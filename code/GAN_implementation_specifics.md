# Generative Adversarial Network 

## Implementation Specifics
### Optimizer 
Adam: ```tf.keras.optimizers.Adam(..., beta=0.5)```

### Vanilla GAN 
**REFRESH**
- feature matching loss
    - Use the Disc to calculate the feature space, given the real and fake images. Therefore, the feature matching loss is as follows ```tf.reduce_mean(tf.abs(real_features - fake_features))```
- unrolled GAN 
    - Train the Discriminator X steps, for every step the Generator is trained

#### Learning Rate Schedule 
```
tf.keras.optimizers.schedules.ExponentialDecay(
    self.lerning_rate, decay_steps=1000, decay_rate=0.95
)
```

#### Discriminator 
##### Label Smooting 
Adding noise to _real_ and _fake_ images, before passed to the Discriminator 
```tf.random.normal(shape=tf.shape(image), stddev=0.1)```
##### Gradient Clipping 
Clipp the gradients of the discriminator, to not make o big steps during updates. 

#### Generator
##### Feature Matching Loss
_Summary_
Feature matching is a reguralization technique, to prevent the generator to train to specifically on the current (of current state/epoch) discriminator. The updated objective requires the generator to create data that matches the feature statistics of real data. Specifically, the generator is required to data that matches the expected value of the features on an intermediate layer of the discriminator. 


Use the Discriminator to deduce features of the real- and fake images. The feature matching loss is the absolute difference between the real features and the fake features.
```
real_features = discriminator(real_images)
fake_features = discriminator(fake_images)
feature_matching_loss = tf.reduce_mean(tf.abs(real_features - fake_features))
```
The generator's loss is then calculated, given the passed loss function - `gen_loss = cross_entropy(discriminator(fake_images))` with the feature matching loss: 
```gen_loss + .1 * feature_matching_loss```
    

#### Noise Vector
Random Normal Distribution 
```tf.random.normal([batch_size, latent_dim])```

### MAD GAN 

