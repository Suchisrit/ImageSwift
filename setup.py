from distutils.core import setup
setup(
  name = 'imageswift',         
  packages = ['imageswift'],   
  version = '0.3',      
  license='MIT',        
  description = 'Easily train an image recognition model on your own images.',   
  author = 'Suchisrit Gangopadhyay',                  
  author_email = 'suchisrit@gmail.com',      
  url = 'https://github.com/Suchisrit/ImageSwift',   
  download_url = 'https://github.com/Suchisrit/ImageSwift/archive/refs/tags/v0.3.tar.gz',  
  keywords = ['CV', 'Image Recognition', 'AI', 'ML'],   
  install_requires=[            
          'tensorflow==2.3.0',
          'seaborn==0.11.2',
          'pandas==1.2.0',
          'matplotlib==3.3.0'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3.8',      
  ],
)