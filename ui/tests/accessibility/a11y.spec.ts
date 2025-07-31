import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';

test.describe('Accessibility Tests', () => {
  test('should meet WCAG 2.1 AA standards on dashboard', async ({ page }) => {
    await page.goto('/');
    
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');
    
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').all();
    const headingLevels = await Promise.all(
      headings.map(heading => heading.evaluate(el => parseInt(el.tagName.charAt(1))))
    );
    
    // Check for proper heading hierarchy (no skipping levels)
    for (let i = 1; i < headingLevels.length; i++) {
      expect(headingLevels[i] - headingLevels[i - 1]).toBeLessThanOrEqual(1);
    }
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/');
    
    // Check for proper ARIA labels on interactive elements
    const buttons = await page.locator('button').all();
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label');
      const textContent = await button.textContent();
      
      // Either aria-label or text content should be present
      expect(ariaLabel || textContent?.trim()).toBeTruthy();
    }
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');
    
    // Test tab navigation
    await page.keyboard.press('Tab');
    const firstFocusable = await page.locator(':focus');
    expect(firstFocusable).toBeTruthy();
    
    // Test tab through all focusable elements
    let focusableCount = 0;
    const maxTabs = 50; // Prevent infinite loops
    
    for (let i = 0; i < maxTabs; i++) {
      await page.keyboard.press('Tab');
      const focused = await page.locator(':focus');
      if (await focused.count() === 0) break;
      focusableCount++;
    }
    
    expect(focusableCount).toBeGreaterThan(0);
  });

  test('should have sufficient color contrast', async ({ page }) => {
    await page.goto('/');
    
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2aa'])
      .analyze();
    
    // Check for color contrast violations
    const contrastViolations = accessibilityScanResults.violations.filter(
      violation => violation.id === 'color-contrast'
    );
    
    expect(contrastViolations).toEqual([]);
  });

  test('should have proper form labels', async ({ page }) => {
    await page.goto('/projects');
    await page.click('text=New Project');
    
    const inputs = await page.locator('input, textarea, select').all();
    for (const input of inputs) {
      const id = await input.getAttribute('id');
      const ariaLabel = await input.getAttribute('aria-label');
      const placeholder = await input.getAttribute('placeholder');
      
      // Each input should have either a label, aria-label, or placeholder
      if (id) {
        const label = await page.locator(`label[for="${id}"]`).count();
        expect(label > 0 || ariaLabel || placeholder).toBeTruthy();
      } else {
        expect(ariaLabel || placeholder).toBeTruthy();
      }
    }
  });

  test('should announce dynamic content changes', async ({ page }) => {
    await page.goto('/');
    
    // Check for live regions
    const liveRegions = await page.locator('[aria-live]').all();
    expect(liveRegions.length).toBeGreaterThan(0);
    
    // Check for status announcements
    const statusElements = await page.locator('[role="status"], [aria-live="polite"]').all();
    expect(statusElements.length).toBeGreaterThan(0);
  });

  test('should have proper focus management', async ({ page }) => {
    await page.goto('/projects');
    
    // Open modal
    await page.click('text=New Project');
    
    // Check that focus is trapped in modal
    await page.keyboard.press('Tab');
    const focusedInModal = await page.locator(':focus');
    const modalContent = await page.locator('[role="dialog"]');
    
    expect(await modalContent.contains(focusedInModal)).toBeTruthy();
  });

  test('should have proper skip links', async ({ page }) => {
    await page.goto('/');
    
    // Check for skip to main content link
    const skipLink = await page.locator('a[href="#main-content"]').count();
    expect(skipLink).toBeGreaterThan(0);
  });

  test('should have proper alt text for images', async ({ page }) => {
    await page.goto('/');
    
    const images = await page.locator('img').all();
    for (const image of images) {
      const alt = await image.getAttribute('alt');
      const ariaLabel = await image.getAttribute('aria-label');
      const role = await image.getAttribute('role');
      
      // Images should have alt text, aria-label, or be decorative
      expect(alt !== null || ariaLabel !== null || role === 'presentation').toBeTruthy();
    }
  });

  test('should have proper table structure', async ({ page }) => {
    await page.goto('/artifacts');
    
    const tables = await page.locator('table').all();
    for (const table of tables) {
      // Check for table headers
      const headers = await table.locator('th').count();
      expect(headers).toBeGreaterThan(0);
      
      // Check for proper scope attributes
      const thElements = await table.locator('th').all();
      for (const th of thElements) {
        const scope = await th.getAttribute('scope');
        expect(scope === 'col' || scope === 'row').toBeTruthy();
      }
    }
  });

  test('should have proper list structure', async ({ page }) => {
    await page.goto('/');
    
    const lists = await page.locator('ul, ol').all();
    for (const list of lists) {
      const listItems = await list.locator('li').count();
      expect(listItems).toBeGreaterThan(0);
    }
  });

  test('should have proper landmark regions', async ({ page }) => {
    await page.goto('/');
    
    // Check for main landmark
    const main = await page.locator('main, [role="main"]').count();
    expect(main).toBeGreaterThan(0);
    
    // Check for navigation landmark
    const nav = await page.locator('nav, [role="navigation"]').count();
    expect(nav).toBeGreaterThan(0);
    
    // Check for banner landmark
    const banner = await page.locator('header, [role="banner"]').count();
    expect(banner).toBeGreaterThan(0);
  });

  test('should have proper error handling', async ({ page }) => {
    await page.goto('/projects');
    
    // Simulate form submission error
    await page.click('text=New Project');
    await page.fill('input[name="name"]', '');
    await page.click('button[type="submit"]');
    
    // Check for error announcement
    const errorMessage = await page.locator('[role="alert"], .error-message').count();
    expect(errorMessage).toBeGreaterThan(0);
  });

  test('should have proper loading states', async ({ page }) => {
    await page.goto('/projects');
    
    // Check for loading indicators
    const loadingElements = await page.locator('[aria-busy="true"], .loading').count();
    expect(loadingElements).toBeGreaterThan(0);
  });

  test('should have proper screen reader support', async ({ page }) => {
    await page.goto('/');
    
    // Check for screen reader only content
    const srOnly = await page.locator('.sr-only, [aria-hidden="true"]').count();
    expect(srOnly).toBeGreaterThan(0);
  });

  test('should have proper focus indicators', async ({ page }) => {
    await page.goto('/');
    
    // Test focus visibility
    await page.keyboard.press('Tab');
    const focused = await page.locator(':focus');
    
    // Check that focused element has visible focus indicator
    const computedStyle = await focused.evaluate(el => {
      const style = window.getComputedStyle(el);
      return {
        outline: style.outline,
        boxShadow: style.boxShadow,
        border: style.border
      };
    });
    
    expect(
      computedStyle.outline !== 'none' ||
      computedStyle.boxShadow !== 'none' ||
      computedStyle.border !== 'none'
    ).toBeTruthy();
  });
}); 