/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: ["@monaco-editor/react"],
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost", pathname: "/**" },
      { protocol: "https", hostname: "api.tinyrl.dev", pathname: "/**" },
    ],
    formats: ["image/webp", "image/avif"],
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  eslint: {
    ignoreDuringBuilds: false,
  },
  compiler: {
    removeConsole: process.env.NODE_ENV === "production",
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    config.module.rules.push({
      test: /\.ttf$/,
      type: "asset/resource",
    });
    return config;
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "origin-when-cross-origin" },
        ],
      },
    ];
  },
  async redirects() {
    return [
      {
        source: "/",
        destination: "/projects",
        permanent: false,
      },
    ];
  },
};

module.exports = nextConfig;
