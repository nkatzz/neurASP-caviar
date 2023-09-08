
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

	orientation(wk2gt_1,125,828). coords(wk2gt_1,115,53,828).  visible(wk2gt_1,828). happensAt(inactive(wk2gt_1),828). orientation(wk2gt_2,142,828). coords(wk2gt_2,99,63,828).  visible(wk2gt_2,828). happensAt(inactive(wk2gt_2),828).
orientation(wk2gt_1,125,829). coords(wk2gt_1,115,53,829).  visible(wk2gt_1,829). happensAt(inactive(wk2gt_1),829). orientation(wk2gt_2,142,829). coords(wk2gt_2,99,63,829).  visible(wk2gt_2,829). happensAt(inactive(wk2gt_2),829).
orientation(wk2gt_1,125,830). coords(wk2gt_1,115,53,830).  visible(wk2gt_1,830). happensAt(inactive(wk2gt_1),830). orientation(wk2gt_2,142,830). coords(wk2gt_2,99,63,830).  visible(wk2gt_2,830). happensAt(inactive(wk2gt_2),830).
orientation(wk2gt_1,125,831). coords(wk2gt_1,115,53,831).  visible(wk2gt_1,831). happensAt(inactive(wk2gt_1),831). orientation(wk2gt_2,142,831). coords(wk2gt_2,99,63,831).  visible(wk2gt_2,831). happensAt(inactive(wk2gt_2),831).
orientation(wk2gt_1,125,832). coords(wk2gt_1,115,53,832).  visible(wk2gt_1,832). happensAt(inactive(wk2gt_1),832). orientation(wk2gt_2,142,832). coords(wk2gt_2,99,63,832).  visible(wk2gt_2,832). happensAt(inactive(wk2gt_2),832).
orientation(wk2gt_1,125,833). coords(wk2gt_1,115,53,833).  visible(wk2gt_1,833). happensAt(inactive(wk2gt_1),833). orientation(wk2gt_2,142,833). coords(wk2gt_2,99,63,833).  visible(wk2gt_2,833). happensAt(inactive(wk2gt_2),833).
orientation(wk2gt_1,125,834). coords(wk2gt_1,115,53,834).  visible(wk2gt_1,834). happensAt(inactive(wk2gt_1),834). orientation(wk2gt_2,142,834). coords(wk2gt_2,99,63,834).  visible(wk2gt_2,834). happensAt(inactive(wk2gt_2),834).
orientation(wk2gt_1,125,835). coords(wk2gt_1,115,53,835).  visible(wk2gt_1,835). happensAt(inactive(wk2gt_1),835). orientation(wk2gt_2,142,835). coords(wk2gt_2,99,63,835).  visible(wk2gt_2,835). happensAt(inactive(wk2gt_2),835).
orientation(wk2gt_1,125,836). coords(wk2gt_1,115,53,836).  visible(wk2gt_1,836). happensAt(inactive(wk2gt_1),836). orientation(wk2gt_2,142,836). coords(wk2gt_2,99,63,836).  visible(wk2gt_2,836). happensAt(inactive(wk2gt_2),836).
orientation(wk2gt_1,125,837). coords(wk2gt_1,115,53,837).  visible(wk2gt_1,837). happensAt(inactive(wk2gt_1),837). orientation(wk2gt_2,142,837). coords(wk2gt_2,99,63,837).  visible(wk2gt_2,837). happensAt(inactive(wk2gt_2),837).
orientation(wk2gt_1,125,838). coords(wk2gt_1,115,53,838).  visible(wk2gt_1,838). happensAt(inactive(wk2gt_1),838). orientation(wk2gt_2,142,838). coords(wk2gt_2,99,63,838).  visible(wk2gt_2,838). happensAt(inactive(wk2gt_2),838).
orientation(wk2gt_1,125,839). coords(wk2gt_1,115,53,839).  visible(wk2gt_1,839). happensAt(inactive(wk2gt_1),839). orientation(wk2gt_2,142,839). coords(wk2gt_2,99,63,839).  visible(wk2gt_2,839). happensAt(inactive(wk2gt_2),839).
orientation(wk2gt_1,125,840). coords(wk2gt_1,115,53,840).  visible(wk2gt_1,840). happensAt(inactive(wk2gt_1),840). orientation(wk2gt_2,142,840). coords(wk2gt_2,99,63,840).  visible(wk2gt_2,840). happensAt(inactive(wk2gt_2),840).
orientation(wk2gt_1,125,841). coords(wk2gt_1,115,53,841).  visible(wk2gt_1,841). happensAt(inactive(wk2gt_1),841). orientation(wk2gt_2,142,841). coords(wk2gt_2,99,63,841).  visible(wk2gt_2,841). happensAt(inactive(wk2gt_2),841).
orientation(wk2gt_1,125,842). coords(wk2gt_1,115,53,842).  visible(wk2gt_1,842). happensAt(inactive(wk2gt_1),842). orientation(wk2gt_2,142,842). coords(wk2gt_2,99,63,842).  visible(wk2gt_2,842). happensAt(inactive(wk2gt_2),842).
orientation(wk2gt_1,125,843). coords(wk2gt_1,115,53,843).  visible(wk2gt_1,843). happensAt(inactive(wk2gt_1),843). orientation(wk2gt_2,142,843). coords(wk2gt_2,99,63,843).  visible(wk2gt_2,843). happensAt(inactive(wk2gt_2),843).
orientation(wk2gt_1,125,844). coords(wk2gt_1,115,53,844).  visible(wk2gt_1,844). happensAt(inactive(wk2gt_1),844). orientation(wk2gt_2,142,844). coords(wk2gt_2,99,63,844).  visible(wk2gt_2,844). happensAt(inactive(wk2gt_2),844).
orientation(wk2gt_1,125,845). coords(wk2gt_1,115,53,845).  visible(wk2gt_1,845). happensAt(inactive(wk2gt_1),845). orientation(wk2gt_2,142,845). coords(wk2gt_2,99,63,845).  visible(wk2gt_2,845). happensAt(inactive(wk2gt_2),845).
orientation(wk2gt_1,125,846). coords(wk2gt_1,115,53,846).  visible(wk2gt_1,846). happensAt(inactive(wk2gt_1),846). orientation(wk2gt_2,142,846). coords(wk2gt_2,99,63,846).  visible(wk2gt_2,846). happensAt(inactive(wk2gt_2),846).
orientation(wk2gt_1,127,847). coords(wk2gt_1,115,53,847).  visible(wk2gt_1,847). happensAt(inactive(wk2gt_1),847). orientation(wk2gt_2,142,847). coords(wk2gt_2,99,63,847).  visible(wk2gt_2,847). happensAt(inactive(wk2gt_2),847).
orientation(wk2gt_1,127,848). coords(wk2gt_1,115,53,848).  visible(wk2gt_1,848). happensAt(inactive(wk2gt_1),848). orientation(wk2gt_2,142,848). coords(wk2gt_2,99,63,848).  visible(wk2gt_2,848). happensAt(inactive(wk2gt_2),848).
orientation(wk2gt_1,127,849). coords(wk2gt_1,115,53,849).  visible(wk2gt_1,849). happensAt(inactive(wk2gt_1),849). orientation(wk2gt_2,142,849). coords(wk2gt_2,99,63,849).  visible(wk2gt_2,849). happensAt(inactive(wk2gt_2),849).
orientation(wk2gt_1,127,850). coords(wk2gt_1,115,53,850).  visible(wk2gt_1,850). happensAt(inactive(wk2gt_1),850). orientation(wk2gt_2,142,850). coords(wk2gt_2,99,63,850).  visible(wk2gt_2,850). happensAt(inactive(wk2gt_2),850).
orientation(wk2gt_1,127,851). coords(wk2gt_1,115,53,851).  visible(wk2gt_1,851). happensAt(inactive(wk2gt_1),851). orientation(wk2gt_2,142,851). coords(wk2gt_2,99,63,851).  visible(wk2gt_2,851). happensAt(inactive(wk2gt_2),851).
orientation(wk2gt_1,127,852). coords(wk2gt_1,115,53,852).  visible(wk2gt_1,852). happensAt(inactive(wk2gt_1),852). orientation(wk2gt_2,142,852). coords(wk2gt_2,99,63,852).  visible(wk2gt_2,852). happensAt(inactive(wk2gt_2),852).
#show holdsAt/2.