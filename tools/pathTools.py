import os

from math import pi, atan2, sin, cos, ceil, sqrt
from datetime import datetime, timedelta
import numpy as np

import scipy.signal as signal
from scipy.signal import savgol_filter
import scipy.io as io

import statsmodels.api as sm
import statsmodels.tsa.stattools as tsatools
from statsmodels.tsa.arima_process import ArmaProcess

# matplotlib-1.5.1
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')
#force matplotlib to use 'GTK3Agg' backend
# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class P2D():

	def __init__(self, t, x, y):

		self.t = t
		self.x = x
		self.y = y

	# +++ relative positions, u:anterior, v:self, w:posterior

	def dist(self, w):
		return np.sqrt((w.x -self.x)**2 +(w.y -self.y)**2)

	def tlag(self, w):
		return (w.t - self.t)

	def speed(self, u, w):
		speed_in  = self.dist(u) /abs(self.t -u.t)
		speed_out = self.dist(w) /abs(w.t -self.t)
		return (speed_in +speed_out) /2

	def turn3(self, u, w):
		t = atan2((w.x - self.x), (w.y - self.y)) - atan2((self.x - u.x), (self.y - u.y))
		if abs(t) > pi:
			if t < 0: t = -2 *pi -t
			else: t = 2 *pi -t
		return t

	def curv3(self, u, w, res, minD):
		# curvature: v.turn3(u, w)/BezierLongitude -> rad/mm (distances are given in mm)
		# limit curvature meassurement to displacements larger than minD microns
		# otherwise assume no movement
		c3 = 0;
		if (self.dist(u) +self.dist(w)):
			# define a Bezier curve
			Bezier = []
			steps = max(ceil((self.dist(u) +self.dist(w)) /res), 5)
			for s in np.linspace(0, 1.0, steps):
				x = (1- s)**2 *u.x +2 *(1 -s) *s *self.x +s**2 *w.x
				y = (1- s)**2 *u.y +2 *(1 -s) *s *self.y +s**2 *w.y
				Bezier.append(P2D(s, x, y))
			D = np.sum([(j.dist(i) +j.dist(k)) for i, j, k in zip(Bezier[:-2], Bezier[1:-1], Bezier[2:])])
			if D >= minD: c3 = self.turn3(u, w) /D
		return c3


class Path():

	# _folder = './leov_gpu/data'
	_folder = './cEl/leov/data'
	_run = 'run_210107_1833.txt'

	def __init__(self, folder = _folder, run = _run, sLag = 31, tLag = 1, scale = 1):

		self.run = '%s/%s' %(folder, run)
		self.raw = []
		self.scale = scale
		self.order = 3
		self.tLag = tLag
		self.sLag = sLag

		if len(run):
			if os.path.exists(self.run):
				# +++ raw-data
				self.all = np.genfromtxt(self.run, delimiter = ',', skip_header = 1)
				# +++ raw-path
				self.raw = [P2D(t, x, y) for t, x, y in zip(self.all[:, 0], self.all[:, 4], self.all[:, 5])]
				print('+++ run %s loaded.' % self.run)
			else:
				print('+++ run %s not found !' % self.run)

	def show(self, start = 0, end = None, points = True):
		# figure
		fig, axs = plt.subplots(figsize = (8, 4.5))
		# raw-path
		X, Y = self.X()[start: end], self.Y()[start: end]
		if (points):
			axs.scatter(X, Y, c = 'r', s = .3)
		axs.plot(X, Y, c = 'b', lw = .5)
		axs.scatter(X[-25:], Y[-25:], c = 'y', s = np.arange(1.0, 26.0, 1.0))
		axs.set_title(self.run)
		plt.show()

	def cut(self, start = 0, end = None, asPath = False):
		if not asPath:
			self.all = self.all[start: end, :]
			self.raw = self.raw[start: end]
		else:
			runCut = Path(run = '')
			runCut.run = self.run
			runCut.all = self.all[start: end, :]
			runCut.raw = self.raw[start: end]
			return runCut

	def T(self):
		return np.array([u.t for u in self.raw])

	def X(self):
		return np.array([u.x for u in self.raw])

	def Y(self):
		return np.array([u.y for u in self.raw])

	def dists(self, tLag = 0):
		if not tLag: tLag = self.tLag
		D = [u.dist(v) for u, v, in zip(self.raw[: -tLag], self.raw[tLag: ])]
		return np.array(D) /self.scale

	def dlength(self):
		return np.sum(self.dists()) /self.tLag

	def tlags(self):
		L = [u.tlag(v) for u, v, in zip(self.raw[: -self.tLag], self.raw[self.tLag: ])]
		return np.array(L)

	def tlength(self):
		return np.sum(self.tlags()) /self.tLag

	def mirror(self, tLag = 0):
		if not tLag: tLag = self.tLag
		return self.raw[1: (tLag +1)][:: -1] + self.raw + self.raw[-(tLag +1): -1][:: -1]

	def speed(self, isSmoothed = False):
		if not isSmoothed:
			sRun = self.smooth(asPath = True)
			R = sRun.mirror()
			S = [v.speed(R[i], R[i +(2 *self.tLag)]) for i, v in enumerate(sRun.raw)]
		else:
			R = self.mirror()
			S = [v.speed(R[i], R[i +(2 *self.tLag)]) for i, v in enumerate(self.raw)]
		return np.array(S) /self.scale

	def turn3(self, isSmoothed = False):
		if not isSmoothed:
			sRun = self.smooth(asPath = True)
			R = sRun.mirror()
			G = [v.turn3(R[i], R[i +(2 *self.tLag)]) for i, v in enumerate(sRun.raw)]
		else:
			R = self.mirror()
			G = [v.turn3(R[i], R[i +(2 *self.tLag)]) for i, v in enumerate(self.raw)]
		return np.array(G)

	def speedHist(self, S, axs, xlabel = True):

		n, bins, patches = axs.hist(x = S, bins = 'auto', color = '#0504aa', alpha = 0.7, rwidth = 0.85)
		axs.grid(axis = 'y', alpha = 0.75)
		if xlabel: axs.set_xlabel(r'$\frac{mm}{s}$')
		# axs.set_ylabel('Frequency')
		# mean speed
		axs.axvline(x = np.mean(S), c = 'r')
		axs.set_title(r'speed,  $\langle s \rangle =$ %6.4f $\frac{mm}{s}$, (tLag %2d)' % (np.mean(S), self.tLag))
		# plt.text(23, 45, r'$\mu=15, b=3$')
		maxfreq = n[1: ].max()
		axs.set_ylim(ymax = np.ceil(maxfreq /100) *100 if maxfreq %100 else maxfreq +100)

	def turnHist(self, G, axs):

		n, bins, patches = axs.hist(x = G, bins = 'auto', color = '#0504aa', alpha = 0.7, rwidth = 0.85)
		axs.grid(axis = 'y', alpha = 0.75)
		axs.set_xlabel(r'$rad$')
		# mean turn
		axs.axvline(x = np.mean(G), c = 'r')
		axs.set_title(r'turn,  $\langle \varphi \rangle =$ %6.4f rad, (tLag %2d)' % (np.mean(G), self.tLag))
		maxfreq = n.max()
		axs.set_ylim(ymax = np.ceil(maxfreq /100) *100 if maxfreq %100 else maxfreq +100)

	def curv3(self, tLag = 2, res = 0.5, minD = 0.010, isSmoothed = False):
		# Att.!!!!!
		# this measure is extremely dependent on the value of tLag and minD;
		# minD represents a tradeoff between accepting large values or considering no curvature
		# by default we use tLag = 2 and minD = 0.005 to smooth out extremly large values;
		# given in rad/mm;
		if not isSmoothed:
			sRun = self.smooth(asPath = True)
			R = sRun.mirror(tLag = tLag)
			K = [v.curv3(R[i], R[i +(2 *tLag)], res = res, minD = minD) for i, v in enumerate(sRun.raw)]
		else:
			R = self.mirror(tLag = tLag)
			K = [v.curv3(R[i], R[i +(2 *tLag)], res = res, minD = minD) for i, v in enumerate(self.raw)]
		return np.array(K)

	def curvPlot(self, tLag = 2, res = 0.5, minD = 0.010, smooth = True, clip = 0):

		K = self.curv3(tLag = tLag, res = res, minD = minD, isSmoothed = not smooth)

		# figure
		fig, axs = plt.subplots(2, 1, figsize = (9, 7))
		plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'x-small'})
		plt.rcParams.update({'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'})
		# plot tSeries
		axs[0].grid(axis = 'y', alpha = 0.75)
		axs[0].set_ylabel(r'$\frac{rad}{mm}$')
		axs[0].set_title(r'curvature $\frac{rad}{mm}$, (tLag %2d)' % tLag)
		axs[0].plot([i for i in range(len(K))], K, c = '#0504aa', lw = .2)

		# histogram
		# clip values below/above clip if given or quantile 0.00001/0.99999 otherwise
		if not clip:
			lowK, uppK = np.quantile(K, [0.01, 0.99])
		else:
			lowK, uppK = -clip, clip
		K[np.where(K < lowK)[0]] = lowK
		K[np.where(K > uppK)[0]] = uppK

		# exclude zero values
		K = K[K != 0]
		# mean values: after-clipping and excluding zero values
		leftMean = np.mean(K[K < 0])
		rightMean = np.mean(K[K > 0])

		lowBrks = np.exp(np.linspace(np.log(1), np.log(abs(lowK) *1.0001 +1), 25)) -1
		uppBrks = np.exp(np.linspace(np.log(1), np.log(abs(uppK) *1.0001 +1), 25)) -1
		brks = np.hstack((-lowBrks[::-1], uppBrks[1: ]))

		n, bins, patches = axs[1].hist(x = K, bins = brks, color = '#0504aa', alpha = 0.7, rwidth = 0.85)
		axs[1].grid(axis = 'y', alpha = 0.75)
		axs[1].set_xlabel(r'$\frac{rad}{mm}$')
		# mean
		axs[1].axvline(x = leftMean, c = 'r', lw = 0.6)
		axs[1].axvline(x = rightMean, c = 'r', lw = 0.6)
		axs[1].set_title(r'curvature,  $\langle \mathcal{K} \rangle =$ %6.4f, %6.4f $\frac{rad}{mm}$, (tLag %2d)' % (leftMean, rightMean, tLag))
		maxfreq = n.max()
		axs[1].set_ylim(ymax = np.ceil(maxfreq /100) *100 if maxfreq %100 else maxfreq +100)
		#
		print('+++ cliping to %6.4f, %6.4f ... leftMean %6.4f, rightMean %6.4f' %(lowK, uppK, leftMean, rightMean))
		plt.show()

	def smooth(self, asPath = True):
		if self.order:
			sX = savgol_filter(self.X(), self.sLag, self.order, mode = 'mirror')
			sY = savgol_filter(self.Y(), self.sLag, self.order, mode = 'mirror')
		else:
			sX = self.X()
			sY = self.Y()
		if asPath:
			smthPath = Path(run = '', scale = self.scale)
			smthPath.raw = [P2D(t, x, y) for t, x, y in zip(self.T(), sX, sY)]
			return smthPath
		else:
			return (sX, sY)

	def startTime(self):
		yy = int('20%s' % self.run[20:22])
		mm = int(self.run[22:24])
		dd = int(self.run[24:26])
		HH = int(self.run[27:29])
		MM = int(self.run[29:31])
		return datetime(year = yy, month = mm, day = dd, hour = HH, minute = MM)

	def endTime(self):
		return self.startTime() + timedelta(seconds = self.tlength(tLag = 1))

	def summary(self, lowQ = 0.001, uppQ = 0.999):

		# sampling cum times
		t = self.T()
		# smoothing window time-width
		tW = self.tlags()

		# smoothed-path
		sPth = self.smooth(asPath = True)

		# path length:
		D = sPth.dists()
		dlen = np.sum(D) /self.tLag
		tlen = sPth.tlength() /60

		# figure
		fig, axs = plt.subplots(2, 3, figsize = (12, 7))
		plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'x-small'})
		plt.rcParams.update({'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'})

		# raw-path
		axs[0, 0].plot(self.X(), self.Y(), c = 'b', lw = .5)
		axs[0, 0].set_title('raw path %4.0f mm' % dlen)
		# sampling cum times
		axs[0, 1].plot(range(t.shape[0]), t, c = 'b', lw = .5)
		axs[0, 1].set_title('cumulated time %4.0f min' % tlen)
		# smoothing window time-width
		axs[0, 2].plot(range(tW.shape[0]), tW, c = 'b', lw = .5)
		axs[0, 2].axhline(np.mean(tW), c = 'r', lw = .7)
		axs[0, 2].set_title('window width %3d, mean_lag_time %5.2f' %(self.tLag, np.mean(tW)))

		# smoothed-path
		axs[1, 0].plot(sPth.X(), sPth.Y(), c = '#0504aa', lw = .5)
		axs[1, 0].set_title('smoothed path, sLag %3d, mean_lag_dist %5.2f' % (self.sLag, np.mean(D)))

		# speed distribution
		S = sPth.speed(isSmoothed = True)
		# lowV, uppV = np.quantile(S, lowQ), np.quantile(S, uppQ)
		# I = np.concatenate((np.where(S > lowV)[0], np.where(S < uppV)[0]))
		# I = np.where(S < uppV)[0]
		sPth.speedHist(S[np.where(S < np.quantile(S, uppQ))[0]], axs[1, 1])
		# the option below is much slower !!!
		# S = [s for s in S if np.quantile(S, lowQ) < s < np.quantile(S, uppQ)]
		# sPth.speedHist(S, axs[1, 1], tLag = self.tLag)

		# turn distribution
		G = sPth.turn3(isSmoothed = True)
		sPth.turnHist(G, axs[1, 2])

		# plot show
		plt.suptitle('+++ %s +++' % self.run)
		plt.show()

	def save(self, fName):
		np.savetxt(fName, self.all, fmt = '%.4f, %.0f, %.0f, %.0f, %.2f, %.2f', delimiter = ',', newline = '\n', header = 'time, frame, camX, camY, centroidX, centroidY')

	def saveSmooth(self, fName):
		sRun = self.smooth(asPath = True)
		smoothRun = np.hstack((self.all, sRun.X()[:, np.newaxis], sRun.Y()[:, np.newaxis], sRun.speed(isSmoothed = True)[:, np.newaxis], sRun.turn3(isSmoothed = True)[:, np.newaxis], sRun.curv3(isSmoothed = True)[:, np.newaxis]))
		np.savetxt(fName, smoothRun, fmt = '%.4f, %.0f, %.0f, %.0f, %.2f, %.2f, %.4f, %.4f, %9.6f, %10.6f, %12.4f', delimiter = ',', newline = '\n', header = 'time, frame, camX, camY, centroidX, centroidY, smoothX, smoothY, speed, turn, curv')

	def lag2dist(self, tLag = 0):
		D = self.dists(tLag = tLag)
		print('+++ tLag %3d, %6.4f mm' %(tLag, np.mean(D)))

	def dist_acf(self, nlags = 60):
		D = self.dists()
		tSeries(D, self, tLag = self.tLag).show(nlags = nlags)

	def speed_acf(self, nlags = 60):
		S = self.speed(isSmoothed = False)
		tSeries(S, self, tLag = self.tLag).show(nlags = nlags)

	def turn_acf(self, nlags = 60):
		G = sPth.turn3(isSmoothed = False)
		tSeries(G, self, tLag = self.tLag).show(nlags = nlags)

	def summary2(self):

		# smoothed-path
		sPth = self.smooth(asPath = True)
		X, Y, T = sPth.X(), sPth.Y(), sPth.tlags()
		S = sPth.speed(isSmoothed = True)

		chunk, cuts = 1, [0]
		for i, cumt in enumerate(np.cumsum(T)):
			if cumt > 3600 *chunk:
				cuts.append(i)
				chunk += 1
		if len(cuts) > 6: cuts = cuts[:6] + [len(sPth.raw)]
		C = ['b', 'c', 'm', 'r', 'y', 'g']

		# figure
		fig, axs = plt.subplots(3, (len(cuts) -1), figsize = ((len(cuts) -1) *2.2, 3 *2.0))
		plt.rcParams.update({'axes.titlesize': 'x-small', 'axes.labelsize': 'x-small'})
		plt.rcParams.update({'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'})

		for i in range(len(cuts) -1):
			# chunk path
			for j in range(i +1):
				a, b = cuts[j], cuts[j +1]
				axs[0, i].plot(X[a: b], Y[a: b], c = C[j], lw = .3)
			axs[0, i].set_title('%4.0f min' % np.round(np.sum(T[0: b]) /60, 0))
			# chunk speed histogram
			a, b = cuts[i], cuts[i +1]
			Sc = S[a: b]
			sPth.speedHist(Sc[np.where(Sc < np.quantile(Sc, 0.999))], axs[1, i], xlabel = False)
			# speed cumulated histogram
			Sc = S[0: b]
			sPth.speedHist(Sc[np.where(Sc < np.quantile(Sc, 0.999))], axs[2, i], xlabel = False)

		# plot show
		plt.tight_layout()
		plt.show()


class tSeries:

	def __init__(self, data, _path, tLag):

		self.X = np.array(data)
		self.path = _path
		a = (tLag -1) //2
		self.t = self.path.T()[a: -a]

	def show(self, nlags = 60):
		# figure
		fig, axs = plt.subplots(2, 2, figsize = (10, 6))
		plt.rcParams.update({'axes.titlesize': 'small'})
		# raw data
		axs[0, 0].plot(self.t, self.X, c = 'b', lw = 0.2)
		axs[0, 0].set_title('raw signal')
		# acf
		acfv = tsatools.acf(self.X, nlags = nlags, fft = True)
		lags = np.arange(nlags)[signal.argrelextrema(acfv, np.greater, axis = 0, order = 1)[0]]
		ttl = 'acf, maxACF lags:'
		print('+++ Signal maxACF lags :')
		for l in lags[ :min(5, len(lags))]:
			ttl += ' %2d' %l
			D = self.path.dists(tLag = l)
			print('   %6.0d, %6.2f' %(l, np.array(D).mean()))
		sm.graphics.tsa.plot_acf(self.X, lags = nlags, ax = axs[1, 0], title = ttl)
		# detrend
		dX = np.diff(self.X, n = 1)
		axs[0, 1].plot(self.t[: -1], dX, c = 'b', lw = 0.2)
		axs[0, 1].set_title('detrend signal (noise)')
		# detrend acf
		acfv = tsatools.acf(dX, nlags = nlags, fft = True)
		lags = np.arange(nlags)[signal.argrelextrema(acfv, np.greater, axis = 0, order = 1)[0]]
		ttl = 'acf, maxACF lags:'
		print('+++ Noise maxACF lags :')
		for l in lags[ :min(5, len(lags))]:
			ttl += ' %2d' %l
			D = self.path.dists(tLag = l)
			print('   %6.0d, %6.2f' %(l, np.array(D).mean()))
		sm.graphics.tsa.plot_acf(dX, lags = nlags, ax = axs[1, 1], title = ttl)
		plt.show()


class wPath(Path):

	def __init__(self, strain = 'N2', run = '2014-05-09_1556', scale = 105):

		self.raw = []
		self.scale = scale
		self.order = 3
		self.tLag = 1
		self.sLag = 1

		self.run = './cEl/Will_Mia/cElPy/data/%s/cMass/%s_mat' % (strain, run)
		if os.path.exists(self.run):
			# +++ raw-data
			txy = io.loadmat('%s/xyReconstr.mat' %self.run)
			self.all = np.array([[t[0], x[0], y[0]] for t, x, y in zip(txy['t'], txy['x'], txy['y'])])
			# +++ raw-path
			self.raw = [P2D(t, x, y) for t, x, y in zip(self.all[:, 0], self.all[:, 1], self.all[:, 2])]
			# +++ Will-Mia smoothed-data
			txy = io.loadmat('%s/xySmooth.mat' %self.run)
			self.wms = np.array([[t[0], x[0], y[0], dx[0], dy[0], dr[0]] for t, x, y, dx, dy, dr in zip(txy['t'], txy['xval'], txy['yval'], txy['dxval'], txy['dyval'], txy['drval'])])
		else:
			print('+++ run %s not found !' %self.run)

		# self.run = './cEl/Will_Mia/cElPy/data/%s/allDta/%s.csv' % (strain, run)
		# if os.path.exists(self.run):
		# 	# +++ Will-Mia smoothed-data
		# 	self.wms = np.genfromtxt(self.run, delimiter = ',', skip_header = 1)
		# else:
		# 	print('+++ run %s not found !' %self.run)

	def summary(self, qntl = 0.999):

		# sampling cum times
		t = self.T()
		# smoothing window time-width
		tW = self.tlags()

		# smoothed-path
		sPth = self.smooth(asPath = True, scale = self.scale)

		# path length:
		D = sPth.dists()
		dlen = np.sum(D) /self.tLag
		tlen = sPth.tlength() /60

		# figure
		fig, axs = plt.subplots(2, 3, figsize = (12, 7))
		plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'x-small'})
		plt.rcParams.update({'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'})

		# raw-path
		axs[0, 0].plot(self.X() /self.scale, self.Y() /self.scale, c = 'b', lw = .5)
		axs[0, 0].set_title('raw path %4.0f mm' % dlen)
		# sampling cum times
		axs[0, 1].plot(range(t.shape[0]), t, c = 'b', lw = .5)
		axs[0, 1].set_title('cumulated time %4.0f min' % tlen)
		# smoothing window time-width
		axs[0, 2].plot(range(tW.shape[0]), tW, c = 'b', lw = .5)
		axs[0, 2].axhline(np.mean(tW), c = 'r', lw = .7)
		axs[0, 2].set_title('window width %3d, mean_lag_time %5.2f' %(self.tLag, np.mean(tW)))

		# smoothed-path versus Will-Mia smoothed-path
		axs[1, 0].plot(sPth.X() /self.scale, sPth.Y() /self.scale, c = '#0504aa', lw = .5)
		axs[1, 0].plot(self.wms[:, 1], self.wms[:, 2], c = '#aa0504', lw = .5)
		axs[1, 0].set_title('smoothed path, sLag %3d, mean_lag_dist %5.2f' % (self.sLag, np.mean(D)))

		# speed distribution
		S = sPth.speed()
		I = np.where(S < np.quantile(S, qntl))[0]
		sPth.speedHist(S[I], axs[1, 1])
		# Will-Mia speed distribution
		S = self.wms[:, 5]
		I = np.where(S < np.quantile(S, qntl))[0]
		self.speedHist(S[I], axs[1, 2])

		# plot show
		plt.suptitle('+++ %s +++' % self.run)
		plt.show()

	def checkX(self, w = 31):
		t = self.T()
		sPth = self.smooth(sLag = w, asPath = True, scale = self.scale)
		plt.plot(t, sPth.X() /sPth.scale, c = '#0504aa', lw = .9)
		plt.plot(t, self.wms[:, 1], c = '#aa0504', lw = .3)
		plt.show()

	def checkdX(self, sLag = 31):
		sPth = self.smooth(sLag = sLag, asPath = True, scale = self.scale)
		a = (sLag -1) //2
		t = sPth.T()[a: -a]
		X = sPth.X() /self.scale
		dX = np.array([(X[i +a] -X[i -a]) for i in range(a, len(X) -a)])
		vX = dX /sPth.tlags(tLag = tLag)
		fig, axs = plt.subplots(2, 1, figsize = (12, 7))
		axs[0].plot(t, vX, c = '#0504aa', lw = .8)
		axs[0].plot(t, self.wms[a: -a, 3], c = '#aa0504', lw = .4)
		axs[1].plot(t, vX -self.wms[a: -a, 3], c = '#0504aa', lw = 0.5)
		plt.show()

	def checkS(self, sLag = 31):
		sPth = self.smooth(sLag = sLag, asPath = True, scale = self.scale)
		a = (sLag -1) //2
		t = sPth.T()[a: -a]
		plt.plot(t, sPth.speed(tLag = tLag), c = '#0504aa', lw = .9)
		plt.plot(t, self.wms[a: -a, 5], c = '#aa0504', lw = .3)
		plt.show()

def mergeRuns(run1, run2):

	deltaTime = run2.startTime() -run1.endTime()
	deltaSecs = run1.all[-1, 0] + deltaTime.seconds + deltaTime.microseconds /10**6
	T = np.array([(t - run2.all[0, 0]) +deltaSecs for t in run2.all[:, 0]])

	lastFrame = run1.all[-1, 1] +1
	F = np.array([(f -run2.all[0, 1]) +lastFrame for f in run2.all[:, 1]])

	merged = Path(run = '')
	merged.run = run1.run
	merged.all = np.vstack((run1.all, np.hstack((T[:, np.newaxis], F[:, np.newaxis], run2.all[:, 2:6]))))
	merged.raw = [P2D(t, x, y) for t, x, y in zip(merged.all[:, 0], merged.all[:, 4], merged.all[:, 5])]

	return merged
