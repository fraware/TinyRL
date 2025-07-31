# TinyRL UI Framework

React-based UI framework for the TinyRL platform, built with Next.js 14, TypeScript, Tailwind CSS, and shadcn/ui components.

## Project Structure

```
ui/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── page.tsx           # Dashboard
│   │   ├── projects/          # Projects page
│   │   ├── pipelines/         # Pipelines page
│   │   ├── artifacts/         # Artifacts page
│   │   ├── fleet/             # Fleet management
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/            # Reusable components
│   │   ├── ui/               # Base UI components (shadcn/ui)
│   │   ├── layout/           # Layout components
│   │   ├── charts/           # Data visualization
│   │   └── projects/         # Project-specific components
│   ├── hooks/                # Custom React hooks
│   ├── lib/                  # Utility functions
│   └── types/                # TypeScript type definitions
├── .storybook/               # Storybook configuration
├── public/                   # Static assets
└── package.json              # Dependencies and scripts
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm, yarn, or pnpm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Development Commands

```bash
# Development
npm run dev              # Start development server
npm run build           # Build for production
npm run start           # Start production server

# Code Quality
npm run lint            # Run ESLint
npm run type-check      # Run TypeScript checks

# Testing
npm run test            # Run Jest tests
npm run test:watch      # Run tests in watch mode
npm run test:coverage   # Generate coverage report

# Storybook
npm run storybook       # Start Storybook dev server
npm run build-storybook # Build Storybook for deployment

# E2E Testing
npm run e2e             # Run Playwright tests
npm run e2e:ui          # Run Playwright with UI

# Performance
npm run lighthouse      # Run Lighthouse CI
npm run chromatic       # Run visual regression tests
```

## Design System

### Color Palette

- **Primary**: Blue (`#3B82F6`) - Main brand color
- **Success**: Green (`#10B981`) - Positive states
- **Warning**: Yellow (`#F59E0B`) - Caution states  
- **Error**: Red (`#EF4444`) - Error states
- **Neutral**: Gray scale for text and backgrounds

### Typography

- **Font**: Inter (Google Fonts)
- **Scale**: Tailwind's default scale with custom adjustments
- **Weights**: 400 (normal), 500 (medium), 600 (semibold), 700 (bold)

### Spacing

- **Base**: 4px grid system
- **Scale**: Tailwind's spacing scale
- **Breakpoints**: Mobile-first responsive design

## Components

### Base Components (shadcn/ui)

- `Button` - Interactive buttons with variants
- `Card` - Content containers with headers
- `Dialog` - Modal dialogs and overlays
- `Input` - Form input fields
- `Badge` - Status indicators and labels
- `Avatar` - User profile images
- `DropdownMenu` - Context menus
- And many more...

### Custom Components

#### Charts
- `SparklineChart` - Minimal line charts for trends
- `RadialGauge` - Circular progress indicators
- `MultiRadialGauge` - Grid of radial gauges

#### Layout
- `Sidebar` - Navigation sidebar with collapsible sections
- `TopNav` - Top navigation bar with search and user menu
- `CommandPalette` - Global command search (⌘+K)

#### Projects
- `ProjectCard` - Project information cards
- `CreateProjectDialog` - Project creation modal
- `ProjectCardSkeleton` - Loading states

## Pages

### Dashboard (`/`)
- Overview of projects, pipelines, and fleet
- Key performance metrics
- Recent activity feed
- Quick action buttons

### Projects (`/projects`)
- Grid/list view of all projects
- Search and filtering
- Project creation workflow
- Status indicators and metrics

### Pipelines (`/pipelines`)
- Pipeline execution monitoring
- Step-by-step progress tracking
- Performance metrics
- Status management

### Artifacts (`/artifacts`)
- Compiled binary management
- Download and deployment
- Verification status
- Hardware requirements

### Fleet (`/fleet`)
- Device fleet management
- Real-time status monitoring
- Performance metrics
- Deployment controls

## Configuration

### Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_SENTRY_DSN=your-sentry-dsn
NEXT_PUBLIC_CHROMATIC_PROJECT_TOKEN=your-chromatic-token
```

### Tailwind Configuration

The project uses a custom Tailwind configuration with:
- Extended color palette
- Custom spacing scale
- Typography plugin
- Container queries plugin
- Custom animations

### Storybook Configuration

- Component documentation
- Interactive controls
- Accessibility testing
- Visual regression testing
- Responsive viewport testing

## Testing

### Unit Testing
- Jest for unit tests
- React Testing Library for component testing
- Coverage reporting

### E2E Testing
- Playwright for end-to-end tests
- Cross-browser testing
- Visual regression testing

### Accessibility Testing
- axe-core integration
- WCAG 2.1 AA compliance
- Keyboard navigation testing
- Screen reader compatibility

## Performance

### Core Web Vitals Targets
- **LCP**: < 2.5s
- **FID**: < 100ms
- **CLS**: < 0.1

### Optimization Strategies
- Code splitting with dynamic imports
- Image optimization with Next.js Image
- Font optimization with next/font
- Bundle analysis and optimization
- Lazy loading for charts and heavy components

## Deployment

### Vercel (Recommended)
```bash
# Deploy to Vercel
vercel --prod
```

### Static Export
```bash
# Build static export
npm run build
npm run export

# Deploy to any static hosting
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Contributing

### Development Workflow

1. **Fork and clone** the repository
2. **Install dependencies** with `npm install`
3. **Create a feature branch** from `main`
4. **Make changes** following the coding standards
5. **Test thoroughly** with unit and e2e tests
6. **Update documentation** as needed
7. **Submit a pull request** with clear description

### Coding Standards

- **TypeScript**: Strict mode enabled
- **ESLint**: Airbnb config with custom rules
- **Prettier**: Consistent code formatting
- **Conventional Commits**: Standardized commit messages

### Component Guidelines

- **Props**: Use TypeScript interfaces for all props
- **Accessibility**: Include ARIA attributes and keyboard support
- **Documentation**: Add JSDoc comments for complex components
- **Testing**: Write unit tests for all components
- **Storybook**: Create stories for visual testing

## Resources

### Documentation
- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [shadcn/ui Documentation](https://ui.shadcn.com)
- [Storybook Documentation](https://storybook.js.org/docs)

### Design Resources
- [Figma Design System](https://figma.com/file/your-design-system)
- [Icon Library](https://lucide.dev)
- [Color Palette](https://coolors.co/your-palette)

### Performance Tools
- [Lighthouse CI](https://github.com/GoogleChrome/lighthouse-ci)
- [WebPageTest](https://www.webpagetest.org)
- [Chromatic](https://www.chromatic.com)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Support

- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the docs folder for detailed guides
- **Community**: Join our Discord server for real-time support

---

Built with ❤️ by the TinyRL Team 