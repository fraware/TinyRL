import * as React from 'react';
import { ImageProps, StaticImageData } from 'next/image';
import { ImageProps as ImageProps$1 } from 'next/legacy/image';

interface StaticRequire {
    default: StaticImageData;
}
declare type StaticImport = StaticRequire | StaticImageData;
declare const ImageContext: React.Context<Partial<Omit<ImageProps, "src"> & {
    src: string | StaticImport;
}> & Omit<ImageProps$1, "src">>;

export { ImageContext };
