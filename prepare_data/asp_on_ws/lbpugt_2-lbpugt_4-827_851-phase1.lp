
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

	orientation(lbpugt_2,86,827). coords(lbpugt_2,292,100,827).  visible(lbpugt_2,827). happensAt(active(lbpugt_2),827). orientation(lbpugt_4,135,827). coords(lbpugt_4,219,34,827).  visible(lbpugt_4,827). happensAt(inactive(lbpugt_4),827).
orientation(lbpugt_2,86,828). coords(lbpugt_2,292,100,828).  visible(lbpugt_2,828). happensAt(active(lbpugt_2),828). orientation(lbpugt_4,135,828). coords(lbpugt_4,219,34,828).  visible(lbpugt_4,828). happensAt(inactive(lbpugt_4),828).
orientation(lbpugt_2,85,829). coords(lbpugt_2,292,100,829).  visible(lbpugt_2,829). happensAt(active(lbpugt_2),829). orientation(lbpugt_4,135,829). coords(lbpugt_4,219,34,829).  visible(lbpugt_4,829). happensAt(inactive(lbpugt_4),829).
orientation(lbpugt_2,85,830). coords(lbpugt_2,292,99,830).  visible(lbpugt_2,830). happensAt(active(lbpugt_2),830). orientation(lbpugt_4,135,830). coords(lbpugt_4,219,34,830).  visible(lbpugt_4,830). happensAt(inactive(lbpugt_4),830).
orientation(lbpugt_2,84,831). coords(lbpugt_2,292,99,831).  visible(lbpugt_2,831). happensAt(active(lbpugt_2),831). orientation(lbpugt_4,135,831). coords(lbpugt_4,219,34,831).  visible(lbpugt_4,831). happensAt(inactive(lbpugt_4),831).
orientation(lbpugt_2,84,832). coords(lbpugt_2,292,99,832).  visible(lbpugt_2,832). happensAt(active(lbpugt_2),832). orientation(lbpugt_4,135,832). coords(lbpugt_4,219,34,832).  visible(lbpugt_4,832). happensAt(inactive(lbpugt_4),832).
orientation(lbpugt_2,84,833). coords(lbpugt_2,292,98,833).  visible(lbpugt_2,833). happensAt(active(lbpugt_2),833). orientation(lbpugt_4,135,833). coords(lbpugt_4,219,34,833).  visible(lbpugt_4,833). happensAt(inactive(lbpugt_4),833).
orientation(lbpugt_2,84,834). coords(lbpugt_2,292,98,834).  visible(lbpugt_2,834). happensAt(active(lbpugt_2),834). orientation(lbpugt_4,135,834). coords(lbpugt_4,219,34,834).  visible(lbpugt_4,834). happensAt(inactive(lbpugt_4),834).
orientation(lbpugt_2,84,835). coords(lbpugt_2,292,97,835).  visible(lbpugt_2,835). happensAt(active(lbpugt_2),835). orientation(lbpugt_4,135,835). coords(lbpugt_4,219,34,835).  visible(lbpugt_4,835). happensAt(inactive(lbpugt_4),835).
orientation(lbpugt_2,84,836). coords(lbpugt_2,292,97,836).  visible(lbpugt_2,836). happensAt(active(lbpugt_2),836). orientation(lbpugt_4,135,836). coords(lbpugt_4,219,34,836).  visible(lbpugt_4,836). happensAt(inactive(lbpugt_4),836).
orientation(lbpugt_2,84,837). coords(lbpugt_2,292,96,837).  visible(lbpugt_2,837). happensAt(active(lbpugt_2),837). orientation(lbpugt_4,135,837). coords(lbpugt_4,219,34,837).  visible(lbpugt_4,837). happensAt(inactive(lbpugt_4),837).
orientation(lbpugt_2,84,838). coords(lbpugt_2,292,96,838).  visible(lbpugt_2,838). happensAt(active(lbpugt_2),838). orientation(lbpugt_4,135,838). coords(lbpugt_4,219,34,838).  visible(lbpugt_4,838). happensAt(inactive(lbpugt_4),838).
orientation(lbpugt_2,84,839). coords(lbpugt_2,292,96,839).  visible(lbpugt_2,839). happensAt(active(lbpugt_2),839). orientation(lbpugt_4,135,839). coords(lbpugt_4,219,34,839).  visible(lbpugt_4,839). happensAt(inactive(lbpugt_4),839).
orientation(lbpugt_2,84,840). coords(lbpugt_2,292,96,840).  visible(lbpugt_2,840). happensAt(active(lbpugt_2),840). orientation(lbpugt_4,135,840). coords(lbpugt_4,219,34,840).  visible(lbpugt_4,840). happensAt(inactive(lbpugt_4),840).
orientation(lbpugt_2,84,841). coords(lbpugt_2,292,96,841).  visible(lbpugt_2,841). happensAt(active(lbpugt_2),841). orientation(lbpugt_4,135,841). coords(lbpugt_4,219,34,841).  visible(lbpugt_4,841). happensAt(inactive(lbpugt_4),841).
orientation(lbpugt_2,84,842). coords(lbpugt_2,292,96,842).  visible(lbpugt_2,842). happensAt(active(lbpugt_2),842). orientation(lbpugt_4,135,842). coords(lbpugt_4,219,34,842).  visible(lbpugt_4,842). happensAt(inactive(lbpugt_4),842).
orientation(lbpugt_2,84,843). coords(lbpugt_2,292,96,843).  visible(lbpugt_2,843). happensAt(active(lbpugt_2),843). orientation(lbpugt_4,135,843). coords(lbpugt_4,219,34,843).  visible(lbpugt_4,843). happensAt(inactive(lbpugt_4),843).
orientation(lbpugt_2,84,844). coords(lbpugt_2,292,96,844).  visible(lbpugt_2,844). happensAt(active(lbpugt_2),844). orientation(lbpugt_4,135,844). coords(lbpugt_4,219,34,844).  visible(lbpugt_4,844). happensAt(inactive(lbpugt_4),844).
orientation(lbpugt_2,84,845). coords(lbpugt_2,292,96,845).  visible(lbpugt_2,845). happensAt(active(lbpugt_2),845). orientation(lbpugt_4,135,845). coords(lbpugt_4,219,34,845).  visible(lbpugt_4,845). happensAt(inactive(lbpugt_4),845).
orientation(lbpugt_2,84,846). coords(lbpugt_2,292,96,846).  visible(lbpugt_2,846). happensAt(active(lbpugt_2),846). orientation(lbpugt_4,135,846). coords(lbpugt_4,219,34,846).  visible(lbpugt_4,846). happensAt(inactive(lbpugt_4),846).
orientation(lbpugt_2,84,847). coords(lbpugt_2,292,96,847).  visible(lbpugt_2,847). happensAt(active(lbpugt_2),847). orientation(lbpugt_4,135,847). coords(lbpugt_4,219,34,847).  visible(lbpugt_4,847). happensAt(inactive(lbpugt_4),847).
orientation(lbpugt_2,84,848). coords(lbpugt_2,292,96,848).  visible(lbpugt_2,848). happensAt(active(lbpugt_2),848). orientation(lbpugt_4,135,848). coords(lbpugt_4,219,34,848).  visible(lbpugt_4,848). happensAt(inactive(lbpugt_4),848).
orientation(lbpugt_2,84,849). coords(lbpugt_2,292,96,849).  visible(lbpugt_2,849). happensAt(active(lbpugt_2),849). orientation(lbpugt_4,135,849). coords(lbpugt_4,219,34,849).  visible(lbpugt_4,849). happensAt(inactive(lbpugt_4),849).
orientation(lbpugt_2,84,850). coords(lbpugt_2,292,96,850).  visible(lbpugt_2,850). happensAt(active(lbpugt_2),850). orientation(lbpugt_4,135,850). coords(lbpugt_4,219,34,850).  visible(lbpugt_4,850). happensAt(inactive(lbpugt_4),850).
orientation(lbpugt_2,84,851). coords(lbpugt_2,292,95,851).  visible(lbpugt_2,851). happensAt(active(lbpugt_2),851). orientation(lbpugt_4,135,851). coords(lbpugt_4,219,34,851).  visible(lbpugt_4,851). happensAt(inactive(lbpugt_4),851).
#show holdsAt/2.