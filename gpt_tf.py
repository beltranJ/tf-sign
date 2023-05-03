

import math
import inspect
from dataclasses import dataclass
import tensorflow as tf
import numpy as np

class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, bias=True, epsilon=1e-5):
        super().__init__()
        self.bias = bias
        self.epsilon = epsilon
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, center=self.bias, scale=self.bias)
        
    def call(self, x):
        return self.layer_norm(x)

class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = tf.keras.layers.Dense(3 * config.n_embd, use_bias=config.bias)
        # output projection
        self.c_proj = tf.keras.layers.Dense(config.n_embd, use_bias=config.bias)
        # regularization
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_dropout)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask_communication_weights = tf.linalg.band_part(tf.ones((config.block_size, config.block_size)), -1, 0)
        self.bias = mask_communication_weights[tf.newaxis, tf.newaxis, :, :]

    def call(self, x):
        shape = tf.shape(x)
        B,T,C = shape[0], shape[1], shape[2] # batch size, sequence length, embedding dimensionality (self.n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = tf.split(self.c_attn(x), 3, axis=2)
        k = tf.transpose(tf.reshape(k, (B, T, self.n_head, C // self.n_head)), (0, 2, 1, 3))
        q = tf.transpose(tf.reshape(q, (B, T, self.n_head, C // self.n_head)), (0, 2, 1, 3))
        v = tf.transpose(tf.reshape(v, (B, T, self.n_head, C // self.n_head)), (0, 2, 1, 3))


        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # normalizing by the hidden dim (embd_num) of each head [C // self.n_head]
        att = tf.matmul(q, k, transpose_b=True) * (1.0 / math.sqrt(k.shape[-1]))
        
        # mask = self.bias[:, :, :T, :T]*tf.ones_like(att) > 0.5
        # If cast leads to an error, use the above mask
        mask = tf.cast(self.bias[:, :, :T, :T], dtype=tf.bool)

        att = tf.where(mask, att, tf.ones_like(att)*float('-inf'))
        
        #(B, nh, T, T) -> (B, nh, T)
        att = tf.nn.softmax(att, axis=-1)

        # dropout on the communication nodes before accessing the values
        att = self.attn_dropout(att)
        y = tf.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, ,nh T, hs)
        y = tf.reshape(tf.transpose(y, (0, 2, 1, 3)), (B, T, C))  # re-assemble all head (B, T, C: self.n_embd)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.c_fc = tf.keras.layers.Dense(4 * config.n_embd, use_bias=config.bias)
        self.c_proj = tf.keras.layers.Dense(config.n_embd, use_bias=config.bias)
        self.dropout = tf.keras.layers.Dropout(config.mlp_dropout)

    def call(self, x):
        x = self.c_fc(x)
        x = tf.keras.activations.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(bias=config.bias)
        self.mlp = MLP(config)

    def call(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    lips_mean: float
    lips_std: float
    left_hands_mean: float
    left_hands_std: float
    pose_mean: float
    pose_std: float
    dims: int = 2
    block_size: int = 50
    landm_size: int = 543
    vocab_size: int = 250
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 100
    lips_start: int = 0
    bias: bool = True
    label_smoothing: float = 0.25
    rotate_hand: bool = False
    rotate_module: int = 10 
    resid_dropout: float = 0.0
    attn_dropout: float = 0.0
    embd_dropout: float = 0.0
    mlp_dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

T=64
C=66
D=2

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
# Activations
GELU = tf.keras.activations.gelu

class LandmarkEmbedding(tf.keras.Model):
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        
         # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def build(self, input_shape):
        pass

    def call(self, x):
        return tf.where(
                # Checks whether landmark is missing in frame
                tf.reduce_sum(x, axis=2, keepdims=True) == 0,
                # If so, the empty embedding is used
                self.empty_embedding,
                # Otherwise the landmark data is embedded
                self.dense(x),
            )
    

class Embedding(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.n_embd = cfg.n_embd
        self.n_embd_lips = cfg.n_embd
        self.n_embd_hand = cfg.n_embd
        self.n_embd_pose = cfg.n_embd
        self.T_hat = cfg.block_size
        self.C = cfg.landm_size
        self.LIPS_START = cfg.lips_start
        self.LIPS_MEAN = cfg.lips_mean
        self.LIPS_STD = cfg.lips_std
        self.LEFT_HANDS_MEAN = cfg.left_hands_mean
        self.LEFT_HANDS_STD =cfg.left_hands_std
        self.POSE_MEAN =cfg.pose_mean
        self.POSE_STD = cfg.pose_std
        self.rotate_hand = cfg.rotate_hand
        self.rotate_module = tf.Variable(cfg.rotate_module, dtype=tf.float32)


        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(self.T_hat+1, self.n_embd, embeddings_initializer=INIT_ZEROS)
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(self.n_embd_lips, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(self.n_embd_hand, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(self.n_embd_pose, 'pose')
        
        # Landmark Weights [lips, left_hand, pose] - learned during training
        self.landmark_weights = tf.Variable(tf.zeros([3], dtype=tf.float32), name='landmark_weights')
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_embd, name='fully_connected_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(GELU),
            tf.keras.layers.Dense(self.n_embd, name='fully_connected_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
        ], name='fc')


        
    def get_diffs(self, l):
        S = l.shape[2]
        other = tf.expand_dims(l, 3)
        other = tf.repeat(other, S, axis=3)
        other = tf.transpose(other, [0,1,3,2])
        diffs = tf.expand_dims(l, 3) - other
        diffs = tf.reshape(diffs, [-1, self.T_hat, S*S])
        return diffs

    def build(self, input_shape):
        pass
 

    def call(self, x, non_empty_frame_idxs, training=False):

        # nans and padded frames were set to zero

        x = tf.slice(x, [0,0,0,0], [-1,self.T_hat, self.C, 2])
        # LIPS
        lips = tf.slice(x, [0,0,self.LIPS_START,0], [-1,self.T_hat, 40, 2])
        lips = tf.where(
                tf.math.equal(lips, 0.0),
                0.0,
                (lips - self.LIPS_MEAN) / self.LIPS_STD,
            )
        # LEFT HAND
        left_hand = tf.slice(x, [0,0,40,0], [-1,self.T_hat, 21, 2])
        left_hand = tf.where(
                tf.math.equal(left_hand, 0.0),
                0.0,
                (left_hand - self.LEFT_HANDS_MEAN) / self.LEFT_HANDS_STD,
            )
        # POSE
        pose = tf.slice(x, [0,0,61,0], [-1,self.T_hat, 5, 2])
        pose = tf.where(
                tf.math.equal(pose, 0.0),
                0.0,
                (pose - self.POSE_MEAN) / self.POSE_STD,
            )
    
        # degrees of rotation to be learned during training        
        
        if self.rotate_hand:
            #rotate left hand coordinates
            angle_deg = tf.random.uniform(shape=(1,), minval=self.rotate_module, maxval=self.rotate_module, dtype=tf.float32)

            r_0 = tf.expand_dims(left_hand[:,:,0,:],2)
            # shape: [10, 64, 1, 2]
            angle_rad = angle_deg * tf.constant(np.pi / 180, dtype=tf.float32)

            cos_angle = tf.cos(angle_rad)
            sin_angle = tf.sin(angle_rad)

            rotation_matrix = tf.reshape(tf.stack([cos_angle, -sin_angle, sin_angle, cos_angle]), (2, 2))


            r_hat = left_hand - r_0

            rotation_matrix = rotation_matrix[tf.newaxis, tf.newaxis, :,:]

            rotated_coords_hat = tf.matmul(r_hat, rotation_matrix)

            left_hand = rotated_coords_hat + r_0
        
        # Flatten
        lips = tf.reshape(lips, [-1, self.T_hat, 40*2])
        left_hand = tf.reshape(left_hand, [-1, self.T_hat, 21*2])
        pose = tf.reshape(pose, [-1, self.T_hat, 5*2])


        # Lips
        lips_embedding = self.lips_embedding(lips)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand)
        # Pose
        pose_embedding = self.pose_embedding(pose)
        # Merge Embeddings of all landmarks with mean pooling (B,T,C,D)
        x = tf.stack((
            lips_embedding, left_hand_embedding, pose_embedding,
        ), axis=3)

        # weighted representation of landmarks
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3) #(B, self.T_hat, self.n_embd)
        # Fully Connected Layers
        x = self.fc(x)
        # Add Positional Embedding
        max_frame_idxs = tf.clip_by_value(
                tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True),
                1,
                np.PINF,
            )
        
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            self.T_hat,
            tf.cast(
                (non_empty_frame_idxs / max_frame_idxs) * self.T_hat,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
        
        return x



class clsr_tsfrm(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

          
        self.embedding = Embedding(self.config)    

        self.transformer = {
            'drop': tf.keras.layers.Dropout(self.config.embd_dropout),
            'h': [Block(self.config) for _ in range(self.config.n_layer)],
            'ln_f': LayerNorm(bias=self.config.bias),
        }


        self.classifier_head = tf.keras.layers.Dense(self.config.vocab_size, use_bias=False)   

            
        

    def build(self, input_shape):
        pass
      
    
    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum([tf.reduce_prod(var.shape).numpy() for var in self.trainable_variables])
        
        # WIth the model changes I'm not certain if we count the pos embedding vector as weights 
        if non_embedding:
            n_params -= tf.reduce_prod(self.transformer['pos_emb'].variables[0].shape).numpy()
        return n_params

    """
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, T, C, D), dtype=tf.float32), 
                                        tf.TensorSpec(shape=(None, T), dtype=tf.int16)])
    """
    def call(self, x_in, non_empty_idxs, targets=None):
        shape = tf.shape(x_in)
        b, t, c, d = shape[0], shape[1], shape[2], shape[3]

        x = self.embedding(x_in, non_empty_idxs)
        x = self.transformer['drop'](x)
        
        for i,block in enumerate(self.transformer['h']):
            x = block(x)

        x = self.transformer['ln_f'](x) # (B, T, self.n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss

            logits = self.classifier_head(x)  # (B, T, self.n_embd) -> (B, T, V)
            # Flatten because SparseCategoricalCrossentropy needs it, X: (B * T, V) and Y: (B * T,)
            if self.config.label_smoothing == 0:
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=-1, reduction=tf.keras.losses.Reduction.NONE)
                #targets expected to be (B, T)
                loss = loss_fn(y_true=targets, y_pred=tf.reshape(logits, (b,t,-1)))
            
            elif self.config.label_smoothing > 0:
                
                y_true_oh = tf.one_hot(targets+1, depth=self.config.vocab_size+1)
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=self.config.label_smoothing,reduction=tf.keras.losses.Reduction.NONE)
                #one hot targets expected to be (B, T, V)
                y_true_oh_ = tf.cast(y_true_oh,dtype=tf.int32)
                loss_smooth = loss_fn(y_true=tf.slice(y_true_oh_, [0, 0, 1], [-1, -1, -1]), y_pred=tf.reshape(logits, (b,t,-1)))
                
                zeros_like_loss_smooth = tf.zeros_like(loss_smooth, dtype=tf.float32)
                loss = tf.where(targets==-1, zeros_like_loss_smooth, loss_smooth)
                

            return logits, loss                
            
        else:

            t_real = tf.reduce_sum(tf.where((tf.reduce_sum(x_in, axis=[-1,-2])>0),1.0,0.0),axis=1)
            last_idxs_real = tf.cast(t_real - 1, tf.int32)
            
            indices = tf.stack([tf.range(b), last_idxs_real], axis=1)
            x_ = tf.gather_nd(x, indices) # (B, self.n_embd)            
            x_ = tf.expand_dims(x_, axis=1) #(B, 1, self.n_embd)
            logits = self.classifier_head(x_) # (B, 1, self.n_embd) -> (B, 1, V) 
            loss = None

        return logits, loss
    



