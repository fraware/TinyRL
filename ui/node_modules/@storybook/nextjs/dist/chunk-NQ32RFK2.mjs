import * as React from 'react';
import { ImageContext as ImageContext$1 } from '@storybook/nextjs/dist/image-context';

var ImageContext=ImageContext$1,ImageDecorator=(Story,{parameters})=>parameters.nextjs?.image?React.createElement(ImageContext.Provider,{value:parameters.nextjs.image},React.createElement(Story,null)):React.createElement(Story,null);

export { ImageDecorator };
