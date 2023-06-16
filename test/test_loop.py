
def test_loop():
    num_devices = jax.device_count()
    t1 = default_timer()
    val_loss = []
    preds = []
    real = []
    for i, inputs in enumerate(dm.test_dataloader()):
            inputs = inputs[:,::2,::2].numpy()
            bs = inputs.shape[0]
            size_x = inputs.shape[1]
            size_y = inputs.shape[2]
            r = np.max([size_x,size_y]) // np.min([size_x,size_y])
            if size_x != size_y:
              l = int(np.min([size_x,size_y]))
            else:
              l = size_x
            for n in range(r):
              nx = int(np.min([size_x, l*(n+1)]))
              ny = int(np.min([size_y, l*(n+1)]))
              inputs = inputs.reshape(num_devices,bs//num_devices,size_x,size_y,output_features,-1)
              outputs = val_step(state,inputs[:,:,nx-l:nx,ny-l:ny],None,None,pred_delta)
              val_loss.append(outputs[0][0])
              preds.append(outputs[1])
              real.append(inputs[:,:,nx-l:nx,ny-l:ny])
              if i == 99:
                break
    t2 = default_timer()
    fig = []
    field_names = ['$\omega$']
    for i in range(output_features):
      fig.append(plot_fields(inputs[0,:,nx-l:nx,ny-l:ny,i],outputs[1][0,:,:,:,i],field_names[i]))
    val_loss = np.mean(np.concatenate(val_loss, axis=0), axis=0)
    print('elapsed time: {} '.format(t2-t1))
    print(val_loss.mean())
    print(preds[0].shape)
    preds = np.concatenate(preds, axis=1)
    real = np.concatenate(real, axis=1)
