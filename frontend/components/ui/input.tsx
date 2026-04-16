import * as React from 'react'
import { cn } from '@/lib/utils'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          'flex h-11 w-full rounded-lg border border-border bg-white px-4 py-3 text-base text-foreground',
          'placeholder:text-foreground-muted',
          'focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent',
          'disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-background-secondary',
          'transition-all duration-150',
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = 'Input'

export { Input }
