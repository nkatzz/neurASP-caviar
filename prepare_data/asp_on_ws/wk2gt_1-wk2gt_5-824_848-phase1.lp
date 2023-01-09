
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

	orientation(wk2gt_1,125,824). coords(wk2gt_1,115,53,824).  visible(wk2gt_1,824). happensAt(inactive(wk2gt_1),824). orientation(wk2gt_5,150,824). coords(wk2gt_5,32,112,824).  visible(wk2gt_5,824). happensAt(walking(wk2gt_5),824).
orientation(wk2gt_1,125,825). coords(wk2gt_1,115,53,825).  visible(wk2gt_1,825). happensAt(inactive(wk2gt_1),825). orientation(wk2gt_5,150,825). coords(wk2gt_5,32,112,825).  visible(wk2gt_5,825). happensAt(walking(wk2gt_5),825).
orientation(wk2gt_1,125,826). coords(wk2gt_1,115,53,826).  visible(wk2gt_1,826). happensAt(inactive(wk2gt_1),826). orientation(wk2gt_5,150,826). coords(wk2gt_5,32,112,826).  visible(wk2gt_5,826). happensAt(walking(wk2gt_5),826).
orientation(wk2gt_1,125,827). coords(wk2gt_1,115,53,827).  visible(wk2gt_1,827). happensAt(inactive(wk2gt_1),827). orientation(wk2gt_5,150,827). coords(wk2gt_5,32,112,827).  visible(wk2gt_5,827). happensAt(walking(wk2gt_5),827).
orientation(wk2gt_1,125,828). coords(wk2gt_1,115,53,828).  visible(wk2gt_1,828). happensAt(inactive(wk2gt_1),828). orientation(wk2gt_5,150,828). coords(wk2gt_5,34,110,828).  visible(wk2gt_5,828). happensAt(walking(wk2gt_5),828).
orientation(wk2gt_1,125,829). coords(wk2gt_1,115,53,829).  visible(wk2gt_1,829). happensAt(inactive(wk2gt_1),829). orientation(wk2gt_5,156,829). coords(wk2gt_5,35,110,829).  visible(wk2gt_5,829). happensAt(walking(wk2gt_5),829).
orientation(wk2gt_1,125,830). coords(wk2gt_1,115,53,830).  visible(wk2gt_1,830). happensAt(inactive(wk2gt_1),830). orientation(wk2gt_5,156,830). coords(wk2gt_5,35,110,830).  visible(wk2gt_5,830). happensAt(walking(wk2gt_5),830).
orientation(wk2gt_1,125,831). coords(wk2gt_1,115,53,831).  visible(wk2gt_1,831). happensAt(inactive(wk2gt_1),831). orientation(wk2gt_5,156,831). coords(wk2gt_5,36,111,831).  visible(wk2gt_5,831). happensAt(walking(wk2gt_5),831).
orientation(wk2gt_1,125,832). coords(wk2gt_1,115,53,832).  visible(wk2gt_1,832). happensAt(inactive(wk2gt_1),832). orientation(wk2gt_5,156,832). coords(wk2gt_5,36,111,832).  visible(wk2gt_5,832). happensAt(walking(wk2gt_5),832).
orientation(wk2gt_1,125,833). coords(wk2gt_1,115,53,833).  visible(wk2gt_1,833). happensAt(inactive(wk2gt_1),833). orientation(wk2gt_5,156,833). coords(wk2gt_5,37,111,833).  visible(wk2gt_5,833). happensAt(walking(wk2gt_5),833).
orientation(wk2gt_1,125,834). coords(wk2gt_1,115,53,834).  visible(wk2gt_1,834). happensAt(inactive(wk2gt_1),834). orientation(wk2gt_5,156,834). coords(wk2gt_5,37,111,834).  visible(wk2gt_5,834). happensAt(walking(wk2gt_5),834).
orientation(wk2gt_1,125,835). coords(wk2gt_1,115,53,835).  visible(wk2gt_1,835). happensAt(inactive(wk2gt_1),835). orientation(wk2gt_5,156,835). coords(wk2gt_5,37,110,835).  visible(wk2gt_5,835). happensAt(walking(wk2gt_5),835).
orientation(wk2gt_1,125,836). coords(wk2gt_1,115,53,836).  visible(wk2gt_1,836). happensAt(inactive(wk2gt_1),836). orientation(wk2gt_5,156,836). coords(wk2gt_5,37,110,836).  visible(wk2gt_5,836). happensAt(walking(wk2gt_5),836).
orientation(wk2gt_1,125,837). coords(wk2gt_1,115,53,837).  visible(wk2gt_1,837). happensAt(inactive(wk2gt_1),837). orientation(wk2gt_5,156,837). coords(wk2gt_5,37,110,837).  visible(wk2gt_5,837). happensAt(walking(wk2gt_5),837).
orientation(wk2gt_1,125,838). coords(wk2gt_1,115,53,838).  visible(wk2gt_1,838). happensAt(inactive(wk2gt_1),838). orientation(wk2gt_5,156,838). coords(wk2gt_5,37,109,838).  visible(wk2gt_5,838). happensAt(walking(wk2gt_5),838).
orientation(wk2gt_1,125,839). coords(wk2gt_1,115,53,839).  visible(wk2gt_1,839). happensAt(inactive(wk2gt_1),839). orientation(wk2gt_5,156,839). coords(wk2gt_5,37,109,839).  visible(wk2gt_5,839). happensAt(walking(wk2gt_5),839).
orientation(wk2gt_1,125,840). coords(wk2gt_1,115,53,840).  visible(wk2gt_1,840). happensAt(inactive(wk2gt_1),840). orientation(wk2gt_5,156,840). coords(wk2gt_5,38,109,840).  visible(wk2gt_5,840). happensAt(walking(wk2gt_5),840).
orientation(wk2gt_1,125,841). coords(wk2gt_1,115,53,841).  visible(wk2gt_1,841). happensAt(inactive(wk2gt_1),841). orientation(wk2gt_5,156,841). coords(wk2gt_5,40,108,841).  visible(wk2gt_5,841). happensAt(walking(wk2gt_5),841).
orientation(wk2gt_1,125,842). coords(wk2gt_1,115,53,842).  visible(wk2gt_1,842). happensAt(inactive(wk2gt_1),842). orientation(wk2gt_5,156,842). coords(wk2gt_5,40,108,842).  visible(wk2gt_5,842). happensAt(walking(wk2gt_5),842).
orientation(wk2gt_1,125,843). coords(wk2gt_1,115,53,843).  visible(wk2gt_1,843). happensAt(inactive(wk2gt_1),843). orientation(wk2gt_5,156,843). coords(wk2gt_5,40,108,843).  visible(wk2gt_5,843). happensAt(walking(wk2gt_5),843).
orientation(wk2gt_1,125,844). coords(wk2gt_1,115,53,844).  visible(wk2gt_1,844). happensAt(inactive(wk2gt_1),844). orientation(wk2gt_5,156,844). coords(wk2gt_5,40,108,844).  visible(wk2gt_5,844). happensAt(walking(wk2gt_5),844).
orientation(wk2gt_1,125,845). coords(wk2gt_1,115,53,845).  visible(wk2gt_1,845). happensAt(inactive(wk2gt_1),845). orientation(wk2gt_5,156,845). coords(wk2gt_5,41,107,845).  visible(wk2gt_5,845). happensAt(walking(wk2gt_5),845).
orientation(wk2gt_1,125,846). coords(wk2gt_1,115,53,846).  visible(wk2gt_1,846). happensAt(inactive(wk2gt_1),846). orientation(wk2gt_5,156,846). coords(wk2gt_5,41,107,846).  visible(wk2gt_5,846). happensAt(walking(wk2gt_5),846).
orientation(wk2gt_1,127,847). coords(wk2gt_1,115,53,847).  visible(wk2gt_1,847). happensAt(inactive(wk2gt_1),847). orientation(wk2gt_5,156,847). coords(wk2gt_5,41,106,847).  visible(wk2gt_5,847). happensAt(walking(wk2gt_5),847).
orientation(wk2gt_1,127,848). coords(wk2gt_1,115,53,848).  visible(wk2gt_1,848). happensAt(inactive(wk2gt_1),848). orientation(wk2gt_5,156,848). coords(wk2gt_5,41,106,848).  visible(wk2gt_5,848). happensAt(walking(wk2gt_5),848).
#show holdsAt/2.