module ANN
  type neuron
    integer:: activation_function_type
    double precision:: stored_value, bias
    LOGICAL:: is_input, is_output
  end type

  type layer
    integer:: order
    type(neuron), dimension(:), allocatable:: neuron_cluster
    double precision, dimension(:,:), allocatable:: weight_values
  end type

  type network
    double precision, dimension(:), allocatable:: layers_composition
    type(layer), dimension(:), allocatable:: neuron_cluster
  end type

  contains
    function setUpNetwork()
    end function

    function setUpLayer()
    end function

    function setUpNeuron()
    end function

    subroutine hello()
      implicit none
      type(neuron), dimension(:), allocatable:: neura
      allocate(neura(10))
        neura(1)%activation_function = 2
        neura(2)%activation_function = 3
        print*, neura(:)%activation_function
      deallocate(neura)
    end subroutine
end module
