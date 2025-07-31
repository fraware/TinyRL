import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test('should navigate to all main pages', async ({ page }) => {
    // Start at the dashboard
    await page.goto('/');
    
    // Check that we're on the dashboard
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    
    // Navigate to projects
    await page.getByRole('link', { name: /projects/i }).click();
    await expect(page.getByRole('heading', { name: 'Projects' })).toBeVisible();
    
    // Navigate to pipelines
    await page.getByRole('link', { name: /pipelines/i }).click();
    await expect(page.getByRole('heading', { name: 'Pipelines' })).toBeVisible();
    
    // Navigate to artifacts
    await page.getByRole('link', { name: /artifacts/i }).click();
    await expect(page.getByRole('heading', { name: 'Artifacts' })).toBeVisible();
    
    // Navigate to fleet
    await page.getByRole('link', { name: /fleet/i }).click();
    await expect(page.getByRole('heading', { name: 'Device Fleet' })).toBeVisible();
  });

  test('should open command palette with keyboard shortcut', async ({ page }) => {
    await page.goto('/');
    
    // Press Cmd+K (or Ctrl+K on Windows/Linux)
    await page.keyboard.press('Meta+K');
    
    // Check that command palette is visible
    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByPlaceholder('Search commands...')).toBeVisible();
    
    // Close with Escape
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).not.toBeVisible();
  });

  test('should search projects', async ({ page }) => {
    await page.goto('/projects');
    
    // Find search input
    const searchInput = page.getByPlaceholder('Search projects...');
    await expect(searchInput).toBeVisible();
    
    // Type in search
    await searchInput.fill('CartPole');
    
    // Check that filtered results are shown
    await expect(page.getByText('CartPole PPO Agent')).toBeVisible();
  });

  test('should create new project', async ({ page }) => {
    await page.goto('/projects');
    
    // Click new project button
    await page.getByRole('button', { name: 'New Project' }).click();
    
    // Check that dialog opens
    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByText('Create New Project')).toBeVisible();
    
    // Fill in project details
    await page.getByLabel('Project Name').fill('Test Project');
    await page.getByLabel('Description').fill('A test project');
    await page.getByLabel('Environment').selectOption('CartPole-v1');
    await page.getByLabel('Algorithm').selectOption('PPO');
    
    // Submit form
    await page.getByRole('button', { name: 'Create Project' }).click();
    
    // Check that dialog closes
    await expect(page.getByRole('dialog')).not.toBeVisible();
  });

  test('should toggle sidebar', async ({ page }) => {
    await page.goto('/');
    
    // Check that sidebar is visible on desktop
    await expect(page.getByRole('navigation')).toBeVisible();
    
    // Click sidebar toggle
    await page.getByRole('button', { name: /toggle sidebar/i }).click();
    
    // Check that sidebar is collapsed
    await expect(page.getByRole('navigation')).toHaveClass(/collapsed/);
  });

  test('should display responsive design on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Check that mobile layout is applied
    await expect(page.getByRole('button', { name: /menu/i })).toBeVisible();
    
    // Open mobile menu
    await page.getByRole('button', { name: /menu/i }).click();
    
    // Check that navigation items are visible
    await expect(page.getByRole('link', { name: /projects/i })).toBeVisible();
  });
}); 