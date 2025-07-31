import { NextConfig } from 'next';

interface SWCLoaderOptions {
    rootDir: string;
    isServer: boolean;
    pagesDir?: string;
    appDir?: string;
    hasReactRefresh: boolean;
    optimizeServerReact?: boolean;
    nextConfig: NextConfig;
    jsConfig: any;
    supportedBrowsers: string[] | undefined;
    swcCacheDir: string;
    serverComponents?: boolean;
    isReactServerLayer?: boolean;
}
declare function pitch(this: any): void;
declare function swcLoader(this: any, inputSource: string, inputSourceMap: any): void;
declare const raw = true;

export { SWCLoaderOptions, swcLoader as default, pitch, raw };
