
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

	orientation(br1gt_2,12,826). coords(br1gt_2,86,209,826).  visible(br1gt_2,826). happensAt(inactive(br1gt_2),826). orientation(br1gt_3,129,826). coords(br1gt_3,280,211,826).  visible(br1gt_3,826). happensAt(walking(br1gt_3),826).
orientation(br1gt_2,12,827). coords(br1gt_2,86,209,827).  visible(br1gt_2,827). happensAt(inactive(br1gt_2),827). orientation(br1gt_3,129,827). coords(br1gt_3,280,212,827).  visible(br1gt_3,827). happensAt(walking(br1gt_3),827).
orientation(br1gt_2,12,828). coords(br1gt_2,86,209,828).  visible(br1gt_2,828). happensAt(inactive(br1gt_2),828). orientation(br1gt_3,129,828). coords(br1gt_3,279,212,828).  visible(br1gt_3,828). happensAt(walking(br1gt_3),828).
orientation(br1gt_2,12,829). coords(br1gt_2,86,209,829).  visible(br1gt_2,829). happensAt(inactive(br1gt_2),829). orientation(br1gt_3,134,829). coords(br1gt_3,278,212,829).  visible(br1gt_3,829). happensAt(walking(br1gt_3),829).
orientation(br1gt_2,12,830). coords(br1gt_2,86,209,830).  visible(br1gt_2,830). happensAt(inactive(br1gt_2),830). orientation(br1gt_3,134,830). coords(br1gt_3,278,212,830).  visible(br1gt_3,830). happensAt(walking(br1gt_3),830).
orientation(br1gt_2,12,831). coords(br1gt_2,86,209,831).  visible(br1gt_2,831). happensAt(inactive(br1gt_2),831). orientation(br1gt_3,134,831). coords(br1gt_3,278,212,831).  visible(br1gt_3,831). happensAt(walking(br1gt_3),831).
orientation(br1gt_2,12,832). coords(br1gt_2,86,209,832).  visible(br1gt_2,832). happensAt(inactive(br1gt_2),832). orientation(br1gt_3,134,832). coords(br1gt_3,279,212,832).  visible(br1gt_3,832). happensAt(walking(br1gt_3),832).
orientation(br1gt_2,12,833). coords(br1gt_2,86,209,833).  visible(br1gt_2,833). happensAt(inactive(br1gt_2),833). orientation(br1gt_3,148,833). coords(br1gt_3,279,212,833).  visible(br1gt_3,833). happensAt(walking(br1gt_3),833).
orientation(br1gt_2,12,834). coords(br1gt_2,86,209,834).  visible(br1gt_2,834). happensAt(inactive(br1gt_2),834). orientation(br1gt_3,157,834). coords(br1gt_3,279,211,834).  visible(br1gt_3,834). happensAt(walking(br1gt_3),834).
orientation(br1gt_2,12,835). coords(br1gt_2,86,209,835).  visible(br1gt_2,835). happensAt(inactive(br1gt_2),835). orientation(br1gt_3,157,835). coords(br1gt_3,279,211,835).  visible(br1gt_3,835). happensAt(walking(br1gt_3),835).
orientation(br1gt_2,12,836). coords(br1gt_2,86,209,836).  visible(br1gt_2,836). happensAt(inactive(br1gt_2),836). orientation(br1gt_3,157,836). coords(br1gt_3,279,209,836).  visible(br1gt_3,836). happensAt(walking(br1gt_3),836).
orientation(br1gt_2,12,837). coords(br1gt_2,86,209,837).  visible(br1gt_2,837). happensAt(inactive(br1gt_2),837). orientation(br1gt_3,157,837). coords(br1gt_3,278,209,837).  visible(br1gt_3,837). happensAt(walking(br1gt_3),837).
orientation(br1gt_2,12,838). coords(br1gt_2,86,209,838).  visible(br1gt_2,838). happensAt(inactive(br1gt_2),838). orientation(br1gt_3,157,838). coords(br1gt_3,278,208,838).  visible(br1gt_3,838). happensAt(walking(br1gt_3),838).
orientation(br1gt_2,12,839). coords(br1gt_2,86,209,839).  visible(br1gt_2,839). happensAt(inactive(br1gt_2),839). orientation(br1gt_3,157,839). coords(br1gt_3,278,208,839).  visible(br1gt_3,839). happensAt(walking(br1gt_3),839).
orientation(br1gt_2,12,840). coords(br1gt_2,86,209,840).  visible(br1gt_2,840). happensAt(inactive(br1gt_2),840). orientation(br1gt_3,157,840). coords(br1gt_3,278,208,840).  visible(br1gt_3,840). happensAt(walking(br1gt_3),840).
orientation(br1gt_2,12,841). coords(br1gt_2,86,209,841).  visible(br1gt_2,841). happensAt(inactive(br1gt_2),841). orientation(br1gt_3,157,841). coords(br1gt_3,278,205,841).  visible(br1gt_3,841). happensAt(walking(br1gt_3),841).
orientation(br1gt_2,12,842). coords(br1gt_2,86,209,842).  visible(br1gt_2,842). happensAt(inactive(br1gt_2),842). orientation(br1gt_3,157,842). coords(br1gt_3,276,204,842).  visible(br1gt_3,842). happensAt(walking(br1gt_3),842).
orientation(br1gt_2,12,843). coords(br1gt_2,86,209,843).  visible(br1gt_2,843). happensAt(inactive(br1gt_2),843). orientation(br1gt_3,157,843). coords(br1gt_3,276,203,843).  visible(br1gt_3,843). happensAt(walking(br1gt_3),843).
orientation(br1gt_2,12,844). coords(br1gt_2,86,209,844).  visible(br1gt_2,844). happensAt(inactive(br1gt_2),844). orientation(br1gt_3,157,844). coords(br1gt_3,277,203,844).  visible(br1gt_3,844). happensAt(walking(br1gt_3),844).
orientation(br1gt_2,12,845). coords(br1gt_2,86,209,845).  visible(br1gt_2,845). happensAt(inactive(br1gt_2),845). orientation(br1gt_3,157,845). coords(br1gt_3,277,203,845).  visible(br1gt_3,845). happensAt(walking(br1gt_3),845).
orientation(br1gt_2,12,846). coords(br1gt_2,86,209,846).  visible(br1gt_2,846). happensAt(inactive(br1gt_2),846). orientation(br1gt_3,157,846). coords(br1gt_3,277,201,846).  visible(br1gt_3,846). happensAt(walking(br1gt_3),846).
orientation(br1gt_2,12,847). coords(br1gt_2,86,209,847).  visible(br1gt_2,847). happensAt(inactive(br1gt_2),847). orientation(br1gt_3,157,847). coords(br1gt_3,277,201,847).  visible(br1gt_3,847). happensAt(walking(br1gt_3),847).
orientation(br1gt_2,12,848). coords(br1gt_2,86,209,848).  visible(br1gt_2,848). happensAt(inactive(br1gt_2),848). orientation(br1gt_3,157,848). coords(br1gt_3,277,199,848).  visible(br1gt_3,848). happensAt(walking(br1gt_3),848).
orientation(br1gt_2,12,849). coords(br1gt_2,86,209,849).  visible(br1gt_2,849). happensAt(inactive(br1gt_2),849). orientation(br1gt_3,157,849). coords(br1gt_3,277,198,849).  visible(br1gt_3,849). happensAt(walking(br1gt_3),849).
orientation(br1gt_2,12,850). coords(br1gt_2,86,209,850).  visible(br1gt_2,850). happensAt(inactive(br1gt_2),850). orientation(br1gt_3,157,850). coords(br1gt_3,278,196,850).  visible(br1gt_3,850). happensAt(walking(br1gt_3),850).
#show holdsAt/2.