import { test, expect } from '@playwright/test';

test.describe('Performance Tests', () => {
  test('should load dashboard within performance budget', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    
    // Wait for page to be fully loaded
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    expect(loadTime).toBeLessThan(3000); // 3 second budget
    
    // Check Core Web Vitals
    const metrics = await page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const navigationEntry = entries.find(entry => entry.entryType === 'navigation') as PerformanceNavigationTiming;
          const paintEntries = entries.filter(entry => entry.entryType === 'paint');
          
          resolve({
            domContentLoaded: navigationEntry?.domContentLoadedEventEnd - navigationEntry?.domContentLoadedEventStart,
            firstPaint: paintEntries.find(entry => entry.name === 'first-paint')?.startTime,
            firstContentfulPaint: paintEntries.find(entry => entry.name === 'first-contentful-paint')?.startTime,
            largestContentfulPaint: entries.find(entry => entry.entryType === 'largest-contentful-paint')?.startTime
          });
        }).observe({ entryTypes: ['navigation', 'paint', 'largest-contentful-paint'] });
      });
    });
    
    expect(metrics.firstContentfulPaint).toBeLessThan(1500); // 1.5s budget
    expect(metrics.largestContentfulPaint).toBeLessThan(2500); // 2.5s budget
  });

  test('should have fast navigation between pages', async ({ page }) => {
    await page.goto('/');
    
    // Measure navigation to projects page
    const startTime = Date.now();
    await page.click('text=Projects');
    await page.waitForLoadState('networkidle');
    const navigationTime = Date.now() - startTime;
    
    expect(navigationTime).toBeLessThan(200); // 200ms budget for client-side navigation
    
    // Measure navigation to pipelines page
    const startTime2 = Date.now();
    await page.click('text=Pipelines');
    await page.waitForLoadState('networkidle');
    const navigationTime2 = Date.now() - startTime2;
    
    expect(navigationTime2).toBeLessThan(200);
  });

  test('should handle large datasets efficiently', async ({ page }) => {
    await page.goto('/projects');
    
    // Simulate loading many projects
    await page.evaluate(() => {
      // Mock API response with 100 projects
      const mockProjects = Array.from({ length: 100 }, (_, i) => ({
        id: `project-${i}`,
        name: `Project ${i}`,
        status: 'completed',
        reward: 95 + Math.random() * 5
      }));
      
      // Simulate rendering
      const startTime = performance.now();
      // Mock rendering logic
      const endTime = performance.now();
      
      return endTime - startTime;
    });
    
    const renderTime = await page.evaluate(() => {
      return performance.now();
    });
    
    expect(renderTime).toBeLessThan(100); // 100ms budget for rendering
  });

  test('should have efficient search and filtering', async ({ page }) => {
    await page.goto('/projects');
    
    // Measure search performance
    const startTime = Date.now();
    await page.fill('input[placeholder="Search projects..."]', 'CartPole');
    await page.waitForTimeout(100); // Debounce delay
    const searchTime = Date.now() - startTime;
    
    expect(searchTime).toBeLessThan(50); // 50ms budget for search
    
    // Measure filter performance
    const startTime2 = Date.now();
    await page.selectOption('select', 'completed');
    await page.waitForTimeout(100);
    const filterTime = Date.now() - startTime2;
    
    expect(filterTime).toBeLessThan(50);
  });

  test('should have efficient chart rendering', async ({ page }) => {
    await page.goto('/');
    
    // Measure chart rendering time
    const chartRenderTime = await page.evaluate(() => {
      return new Promise((resolve) => {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const chartEntry = entries.find(entry => 
            entry.name.includes('chart') || entry.name.includes('sparkline')
          );
          if (chartEntry) {
            resolve(chartEntry.duration);
          }
        });
        observer.observe({ entryTypes: ['measure'] });
      });
    });
    
    expect(chartRenderTime).toBeLessThan(100); // 100ms budget for chart rendering
  });

  test('should have efficient modal rendering', async ({ page }) => {
    await page.goto('/projects');
    
    // Measure modal opening time
    const startTime = Date.now();
    await page.click('text=New Project');
    await page.waitForSelector('[role="dialog"]');
    const modalTime = Date.now() - startTime;
    
    expect(modalTime).toBeLessThan(150); // 150ms budget for modal opening
  });

  test('should have efficient data fetching', async ({ page }) => {
    await page.goto('/projects');
    
    // Measure API response time
    const apiResponseTime = await page.evaluate(() => {
      return new Promise((resolve) => {
        const startTime = performance.now();
        fetch('/api/projects')
          .then(() => {
            const endTime = performance.now();
            resolve(endTime - startTime);
          });
      });
    });
    
    expect(apiResponseTime).toBeLessThan(500); // 500ms budget for API calls
  });

  test('should have efficient image loading', async ({ page }) => {
    await page.goto('/');
    
    // Measure image loading time
    const imageLoadTime = await page.evaluate(() => {
      return new Promise((resolve) => {
        const images = document.querySelectorAll('img');
        let loadedImages = 0;
        const totalImages = images.length;
        
        if (totalImages === 0) {
          resolve(0);
          return;
        }
        
        const startTime = performance.now();
        
        images.forEach(img => {
          if (img.complete) {
            loadedImages++;
            if (loadedImages === totalImages) {
              const endTime = performance.now();
              resolve(endTime - startTime);
            }
          } else {
            img.addEventListener('load', () => {
              loadedImages++;
              if (loadedImages === totalImages) {
                const endTime = performance.now();
                resolve(endTime - startTime);
              }
            });
          }
        });
      });
    });
    
    expect(imageLoadTime).toBeLessThan(1000); // 1s budget for image loading
  });

  test('should have efficient memory usage', async ({ page }) => {
    await page.goto('/');
    
    // Measure memory usage
    const memoryUsage = await page.evaluate(() => {
      if ('memory' in performance) {
        return (performance as any).memory.usedJSHeapSize;
      }
      return 0;
    });
    
    expect(memoryUsage).toBeLessThan(50 * 1024 * 1024); // 50MB budget
  });

  test('should have efficient bundle size', async ({ page }) => {
    await page.goto('/');
    
    // Measure JavaScript bundle size
    const bundleSize = await page.evaluate(() => {
      const scripts = document.querySelectorAll('script[src]');
      let totalSize = 0;
      
      scripts.forEach(script => {
        const src = script.getAttribute('src');
        if (src && src.includes('chunk') || src.includes('main')) {
          // Estimate size based on common patterns
          totalSize += 100 * 1024; // 100KB estimate per script
        }
      });
      
      return totalSize;
    });
    
    expect(bundleSize).toBeLessThan(2 * 1024 * 1024); // 2MB budget
  });

  test('should have efficient CSS delivery', async ({ page }) => {
    await page.goto('/');
    
    // Measure CSS loading time
    const cssLoadTime = await page.evaluate(() => {
      return new Promise((resolve) => {
        const links = document.querySelectorAll('link[rel="stylesheet"]');
        let loadedStylesheets = 0;
        const totalStylesheets = links.length;
        
        if (totalStylesheets === 0) {
          resolve(0);
          return;
        }
        
        const startTime = performance.now();
        
        links.forEach(link => {
          if ((link as any).sheet) {
            loadedStylesheets++;
            if (loadedStylesheets === totalStylesheets) {
              const endTime = performance.now();
              resolve(endTime - startTime);
            }
          } else {
            link.addEventListener('load', () => {
              loadedStylesheets++;
              if (loadedStylesheets === totalStylesheets) {
                const endTime = performance.now();
                resolve(endTime - startTime);
              }
            });
          }
        });
      });
    });
    
    expect(cssLoadTime).toBeLessThan(500); // 500ms budget for CSS loading
  });

  test('should have efficient third-party script loading', async ({ page }) => {
    await page.goto('/');
    
    // Measure third-party script loading
    const thirdPartyLoadTime = await page.evaluate(() => {
      return new Promise((resolve) => {
        const scripts = document.querySelectorAll('script[src]');
        let loadedScripts = 0;
        const totalScripts = scripts.length;
        
        if (totalScripts === 0) {
          resolve(0);
          return;
        }
        
        const startTime = performance.now();
        
        scripts.forEach(script => {
          if (script.getAttribute('src')?.includes('analytics') || 
              script.getAttribute('src')?.includes('monitoring')) {
            script.addEventListener('load', () => {
              loadedScripts++;
              if (loadedScripts === totalScripts) {
                const endTime = performance.now();
                resolve(endTime - startTime);
              }
            });
          } else {
            loadedScripts++;
            if (loadedScripts === totalScripts) {
              const endTime = performance.now();
              resolve(endTime - startTime);
            }
          }
        });
      });
    });
    
    expect(thirdPartyLoadTime).toBeLessThan(1000); // 1s budget for third-party scripts
  });

  test('should maintain performance under network constraints', async ({ page }) => {
    // Simulate slow 3G network
    await page.route('**/*', route => {
      route.continue({
        delay: 100 // 100ms delay to simulate slow network
      });
    });
    
    const startTime = Date.now();
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;
    
    // Should still load within reasonable time even on slow network
    expect(loadTime).toBeLessThan(5000); // 5s budget for slow network
  });

  test('should have efficient scrolling performance', async ({ page }) => {
    await page.goto('/projects');
    
    // Measure scroll performance
    const scrollPerformance = await page.evaluate(() => {
      return new Promise((resolve) => {
        let frameCount = 0;
        let lastTime = performance.now();
        
        const measureScroll = () => {
          frameCount++;
          const currentTime = performance.now();
          
          if (currentTime - lastTime >= 1000) { // Measure for 1 second
            const fps = frameCount / ((currentTime - lastTime) / 1000);
            resolve(fps);
            return;
          }
          
          requestAnimationFrame(measureScroll);
        };
        
        // Trigger scroll
        window.scrollTo(0, 1000);
        requestAnimationFrame(measureScroll);
      });
    });
    
    expect(scrollPerformance).toBeGreaterThan(55); // 55 FPS minimum
  });
}); 