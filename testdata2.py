def outer_function():
    print("Entering outer function")

    def middle_function():
        print("  Entering middle function")

        def inner_function():
            print("    Entering inner function")
            for i in range(3):
                print(f"      Loop iteration {i}")
            print("    Exiting inner function")

        print("middle func completed")

    # sub_process()
    print("outer func finished")


def outer_function2():
    print("lallala")

    def middle_function2():
        print("  ouuuuu")

# Call the main process
outer_function()
print("hello world")

