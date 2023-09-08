
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(wk2gt_2,142,824). coords(wk2gt_2,99,63,824).  visible(wk2gt_2,824). happensAt(inactive(wk2gt_2),824). orientation(wk2gt_5,150,824). coords(wk2gt_5,32,112,824).  visible(wk2gt_5,824). happensAt(walking(wk2gt_5),824).
orientation(wk2gt_2,142,825). coords(wk2gt_2,99,63,825).  visible(wk2gt_2,825). happensAt(inactive(wk2gt_2),825). orientation(wk2gt_5,150,825). coords(wk2gt_5,32,112,825).  visible(wk2gt_5,825). happensAt(walking(wk2gt_5),825).
orientation(wk2gt_2,142,826). coords(wk2gt_2,99,63,826).  visible(wk2gt_2,826). happensAt(inactive(wk2gt_2),826). orientation(wk2gt_5,150,826). coords(wk2gt_5,32,112,826).  visible(wk2gt_5,826). happensAt(walking(wk2gt_5),826).
orientation(wk2gt_2,142,827). coords(wk2gt_2,99,63,827).  visible(wk2gt_2,827). happensAt(inactive(wk2gt_2),827). orientation(wk2gt_5,150,827). coords(wk2gt_5,32,112,827).  visible(wk2gt_5,827). happensAt(walking(wk2gt_5),827).
orientation(wk2gt_2,142,828). coords(wk2gt_2,99,63,828).  visible(wk2gt_2,828). happensAt(inactive(wk2gt_2),828). orientation(wk2gt_5,150,828). coords(wk2gt_5,34,110,828).  visible(wk2gt_5,828). happensAt(walking(wk2gt_5),828).
orientation(wk2gt_2,142,829). coords(wk2gt_2,99,63,829).  visible(wk2gt_2,829). happensAt(inactive(wk2gt_2),829). orientation(wk2gt_5,156,829). coords(wk2gt_5,35,110,829).  visible(wk2gt_5,829). happensAt(walking(wk2gt_5),829).
orientation(wk2gt_2,142,830). coords(wk2gt_2,99,63,830).  visible(wk2gt_2,830). happensAt(inactive(wk2gt_2),830). orientation(wk2gt_5,156,830). coords(wk2gt_5,35,110,830).  visible(wk2gt_5,830). happensAt(walking(wk2gt_5),830).
orientation(wk2gt_2,142,831). coords(wk2gt_2,99,63,831).  visible(wk2gt_2,831). happensAt(inactive(wk2gt_2),831). orientation(wk2gt_5,156,831). coords(wk2gt_5,36,111,831).  visible(wk2gt_5,831). happensAt(walking(wk2gt_5),831).
orientation(wk2gt_2,142,832). coords(wk2gt_2,99,63,832).  visible(wk2gt_2,832). happensAt(inactive(wk2gt_2),832). orientation(wk2gt_5,156,832). coords(wk2gt_5,36,111,832).  visible(wk2gt_5,832). happensAt(walking(wk2gt_5),832).
orientation(wk2gt_2,142,833). coords(wk2gt_2,99,63,833).  visible(wk2gt_2,833). happensAt(inactive(wk2gt_2),833). orientation(wk2gt_5,156,833). coords(wk2gt_5,37,111,833).  visible(wk2gt_5,833). happensAt(walking(wk2gt_5),833).
orientation(wk2gt_2,142,834). coords(wk2gt_2,99,63,834).  visible(wk2gt_2,834). happensAt(inactive(wk2gt_2),834). orientation(wk2gt_5,156,834). coords(wk2gt_5,37,111,834).  visible(wk2gt_5,834). happensAt(walking(wk2gt_5),834).
orientation(wk2gt_2,142,835). coords(wk2gt_2,99,63,835).  visible(wk2gt_2,835). happensAt(inactive(wk2gt_2),835). orientation(wk2gt_5,156,835). coords(wk2gt_5,37,110,835).  visible(wk2gt_5,835). happensAt(walking(wk2gt_5),835).
orientation(wk2gt_2,142,836). coords(wk2gt_2,99,63,836).  visible(wk2gt_2,836). happensAt(inactive(wk2gt_2),836). orientation(wk2gt_5,156,836). coords(wk2gt_5,37,110,836).  visible(wk2gt_5,836). happensAt(walking(wk2gt_5),836).
orientation(wk2gt_2,142,837). coords(wk2gt_2,99,63,837).  visible(wk2gt_2,837). happensAt(inactive(wk2gt_2),837). orientation(wk2gt_5,156,837). coords(wk2gt_5,37,110,837).  visible(wk2gt_5,837). happensAt(walking(wk2gt_5),837).
orientation(wk2gt_2,142,838). coords(wk2gt_2,99,63,838).  visible(wk2gt_2,838). happensAt(inactive(wk2gt_2),838). orientation(wk2gt_5,156,838). coords(wk2gt_5,37,109,838).  visible(wk2gt_5,838). happensAt(walking(wk2gt_5),838).
orientation(wk2gt_2,142,839). coords(wk2gt_2,99,63,839).  visible(wk2gt_2,839). happensAt(inactive(wk2gt_2),839). orientation(wk2gt_5,156,839). coords(wk2gt_5,37,109,839).  visible(wk2gt_5,839). happensAt(walking(wk2gt_5),839).
orientation(wk2gt_2,142,840). coords(wk2gt_2,99,63,840).  visible(wk2gt_2,840). happensAt(inactive(wk2gt_2),840). orientation(wk2gt_5,156,840). coords(wk2gt_5,38,109,840).  visible(wk2gt_5,840). happensAt(walking(wk2gt_5),840).
orientation(wk2gt_2,142,841). coords(wk2gt_2,99,63,841).  visible(wk2gt_2,841). happensAt(inactive(wk2gt_2),841). orientation(wk2gt_5,156,841). coords(wk2gt_5,40,108,841).  visible(wk2gt_5,841). happensAt(walking(wk2gt_5),841).
orientation(wk2gt_2,142,842). coords(wk2gt_2,99,63,842).  visible(wk2gt_2,842). happensAt(inactive(wk2gt_2),842). orientation(wk2gt_5,156,842). coords(wk2gt_5,40,108,842).  visible(wk2gt_5,842). happensAt(walking(wk2gt_5),842).
orientation(wk2gt_2,142,843). coords(wk2gt_2,99,63,843).  visible(wk2gt_2,843). happensAt(inactive(wk2gt_2),843). orientation(wk2gt_5,156,843). coords(wk2gt_5,40,108,843).  visible(wk2gt_5,843). happensAt(walking(wk2gt_5),843).
orientation(wk2gt_2,142,844). coords(wk2gt_2,99,63,844).  visible(wk2gt_2,844). happensAt(inactive(wk2gt_2),844). orientation(wk2gt_5,156,844). coords(wk2gt_5,40,108,844).  visible(wk2gt_5,844). happensAt(walking(wk2gt_5),844).
orientation(wk2gt_2,142,845). coords(wk2gt_2,99,63,845).  visible(wk2gt_2,845). happensAt(inactive(wk2gt_2),845). orientation(wk2gt_5,156,845). coords(wk2gt_5,41,107,845).  visible(wk2gt_5,845). happensAt(walking(wk2gt_5),845).
orientation(wk2gt_2,142,846). coords(wk2gt_2,99,63,846).  visible(wk2gt_2,846). happensAt(inactive(wk2gt_2),846). orientation(wk2gt_5,156,846). coords(wk2gt_5,41,107,846).  visible(wk2gt_5,846). happensAt(walking(wk2gt_5),846).
orientation(wk2gt_2,142,847). coords(wk2gt_2,99,63,847).  visible(wk2gt_2,847). happensAt(inactive(wk2gt_2),847). orientation(wk2gt_5,156,847). coords(wk2gt_5,41,106,847).  visible(wk2gt_5,847). happensAt(walking(wk2gt_5),847).
orientation(wk2gt_2,142,848). coords(wk2gt_2,99,63,848).  visible(wk2gt_2,848). happensAt(inactive(wk2gt_2),848). orientation(wk2gt_5,156,848). coords(wk2gt_5,41,106,848).  visible(wk2gt_5,848). happensAt(walking(wk2gt_5),848).
#show holdsAt/2.