
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

	orientation(lbgt_3,61,823). coords(lbgt_3,146,162,823).  visible(lbgt_3,823). happensAt(inactive(lbgt_3),823). orientation(lbgt_4,136,823). coords(lbgt_4,82,82,823).  visible(lbgt_4,823). happensAt(walking(lbgt_4),823).
orientation(lbgt_3,61,824). coords(lbgt_3,146,162,824).  visible(lbgt_3,824). happensAt(inactive(lbgt_3),824). orientation(lbgt_4,136,824). coords(lbgt_4,82,82,824).  visible(lbgt_4,824). happensAt(walking(lbgt_4),824).
orientation(lbgt_3,61,825). coords(lbgt_3,146,162,825).  visible(lbgt_3,825). happensAt(inactive(lbgt_3),825). orientation(lbgt_4,136,825). coords(lbgt_4,82,82,825).  visible(lbgt_4,825). happensAt(walking(lbgt_4),825).
orientation(lbgt_3,61,826). coords(lbgt_3,146,162,826).  visible(lbgt_3,826). happensAt(inactive(lbgt_3),826). orientation(lbgt_4,136,826). coords(lbgt_4,82,82,826).  visible(lbgt_4,826). happensAt(walking(lbgt_4),826).
orientation(lbgt_3,61,827). coords(lbgt_3,146,162,827).  visible(lbgt_3,827). happensAt(inactive(lbgt_3),827). orientation(lbgt_4,136,827). coords(lbgt_4,82,81,827).  visible(lbgt_4,827). happensAt(walking(lbgt_4),827).
orientation(lbgt_3,61,828). coords(lbgt_3,146,162,828).  visible(lbgt_3,828). happensAt(inactive(lbgt_3),828). orientation(lbgt_4,136,828). coords(lbgt_4,82,80,828).  visible(lbgt_4,828). happensAt(walking(lbgt_4),828).
orientation(lbgt_3,61,829). coords(lbgt_3,146,162,829).  visible(lbgt_3,829). happensAt(inactive(lbgt_3),829). orientation(lbgt_4,136,829). coords(lbgt_4,82,79,829).  visible(lbgt_4,829). happensAt(walking(lbgt_4),829).
orientation(lbgt_3,61,830). coords(lbgt_3,146,162,830).  visible(lbgt_3,830). happensAt(inactive(lbgt_3),830). orientation(lbgt_4,136,830). coords(lbgt_4,82,79,830).  visible(lbgt_4,830). happensAt(walking(lbgt_4),830).
orientation(lbgt_3,61,831). coords(lbgt_3,146,162,831).  visible(lbgt_3,831). happensAt(inactive(lbgt_3),831). orientation(lbgt_4,136,831). coords(lbgt_4,82,78,831).  visible(lbgt_4,831). happensAt(walking(lbgt_4),831).
orientation(lbgt_3,61,832). coords(lbgt_3,146,162,832).  visible(lbgt_3,832). happensAt(inactive(lbgt_3),832). orientation(lbgt_4,136,832). coords(lbgt_4,82,76,832).  visible(lbgt_4,832). happensAt(walking(lbgt_4),832).
orientation(lbgt_3,61,833). coords(lbgt_3,146,162,833).  visible(lbgt_3,833). happensAt(inactive(lbgt_3),833). orientation(lbgt_4,136,833). coords(lbgt_4,82,75,833).  visible(lbgt_4,833). happensAt(walking(lbgt_4),833).
orientation(lbgt_3,61,834). coords(lbgt_3,146,162,834).  visible(lbgt_3,834). happensAt(inactive(lbgt_3),834). orientation(lbgt_4,136,834). coords(lbgt_4,82,75,834).  visible(lbgt_4,834). happensAt(walking(lbgt_4),834).
orientation(lbgt_3,61,835). coords(lbgt_3,146,162,835).  visible(lbgt_3,835). happensAt(inactive(lbgt_3),835). orientation(lbgt_4,136,835). coords(lbgt_4,82,75,835).  visible(lbgt_4,835). happensAt(walking(lbgt_4),835).
orientation(lbgt_3,61,836). coords(lbgt_3,146,162,836).  visible(lbgt_3,836). happensAt(inactive(lbgt_3),836). orientation(lbgt_4,136,836). coords(lbgt_4,82,74,836).  visible(lbgt_4,836). happensAt(walking(lbgt_4),836).
orientation(lbgt_3,61,837). coords(lbgt_3,146,162,837).  visible(lbgt_3,837). happensAt(inactive(lbgt_3),837). orientation(lbgt_4,136,837). coords(lbgt_4,82,74,837).  visible(lbgt_4,837). happensAt(walking(lbgt_4),837).
orientation(lbgt_3,61,838). coords(lbgt_3,146,162,838).  visible(lbgt_3,838). happensAt(inactive(lbgt_3),838). orientation(lbgt_4,136,838). coords(lbgt_4,82,73,838).  visible(lbgt_4,838). happensAt(walking(lbgt_4),838).
orientation(lbgt_3,61,839). coords(lbgt_3,146,162,839).  visible(lbgt_3,839). happensAt(inactive(lbgt_3),839). orientation(lbgt_4,136,839). coords(lbgt_4,82,73,839).  visible(lbgt_4,839). happensAt(walking(lbgt_4),839).
orientation(lbgt_3,61,840). coords(lbgt_3,146,162,840).  visible(lbgt_3,840). happensAt(inactive(lbgt_3),840). orientation(lbgt_4,136,840). coords(lbgt_4,82,72,840).  visible(lbgt_4,840). happensAt(walking(lbgt_4),840).
orientation(lbgt_3,61,841). coords(lbgt_3,146,162,841).  visible(lbgt_3,841). happensAt(inactive(lbgt_3),841). orientation(lbgt_4,136,841). coords(lbgt_4,82,72,841).  visible(lbgt_4,841). happensAt(walking(lbgt_4),841).
orientation(lbgt_3,61,842). coords(lbgt_3,146,162,842).  visible(lbgt_3,842). happensAt(inactive(lbgt_3),842). orientation(lbgt_4,136,842). coords(lbgt_4,82,71,842).  visible(lbgt_4,842). happensAt(walking(lbgt_4),842).
orientation(lbgt_3,61,843). coords(lbgt_3,146,162,843).  visible(lbgt_3,843). happensAt(inactive(lbgt_3),843). orientation(lbgt_4,136,843). coords(lbgt_4,82,69,843).  visible(lbgt_4,843). happensAt(walking(lbgt_4),843).
orientation(lbgt_3,61,844). coords(lbgt_3,146,162,844).  visible(lbgt_3,844). happensAt(inactive(lbgt_3),844). orientation(lbgt_4,136,844). coords(lbgt_4,83,68,844).  visible(lbgt_4,844). happensAt(walking(lbgt_4),844).
orientation(lbgt_3,61,845). coords(lbgt_3,146,162,845).  visible(lbgt_3,845). happensAt(inactive(lbgt_3),845). orientation(lbgt_4,136,845). coords(lbgt_4,84,67,845).  visible(lbgt_4,845). happensAt(walking(lbgt_4),845).
orientation(lbgt_3,61,846). coords(lbgt_3,146,162,846).  visible(lbgt_3,846). happensAt(inactive(lbgt_3),846). orientation(lbgt_4,136,846). coords(lbgt_4,84,67,846).  visible(lbgt_4,846). happensAt(walking(lbgt_4),846).
orientation(lbgt_3,61,847). coords(lbgt_3,146,162,847).  visible(lbgt_3,847). happensAt(inactive(lbgt_3),847). orientation(lbgt_4,136,847). coords(lbgt_4,85,67,847).  visible(lbgt_4,847). happensAt(walking(lbgt_4),847).
#show holdsAt/2.