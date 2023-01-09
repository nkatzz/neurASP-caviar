
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

	orientation(lb2gt_5,152,831). coords(lb2gt_5,108,72,831).  visible(lb2gt_5,831). happensAt(walking(lb2gt_5),831). orientation(lb2gt_6,151,831). coords(lb2gt_6,72,122,831).  visible(lb2gt_6,831). happensAt(inactive(lb2gt_6),831).
orientation(lb2gt_5,152,832). coords(lb2gt_5,107,73,832).  visible(lb2gt_5,832). happensAt(walking(lb2gt_5),832). orientation(lb2gt_6,151,832). coords(lb2gt_6,72,122,832).  visible(lb2gt_6,832). happensAt(inactive(lb2gt_6),832).
orientation(lb2gt_5,152,833). coords(lb2gt_5,106,75,833).  visible(lb2gt_5,833). happensAt(walking(lb2gt_5),833). orientation(lb2gt_6,151,833). coords(lb2gt_6,72,122,833).  visible(lb2gt_6,833). happensAt(inactive(lb2gt_6),833).
orientation(lb2gt_5,152,834). coords(lb2gt_5,106,76,834).  visible(lb2gt_5,834). happensAt(walking(lb2gt_5),834). orientation(lb2gt_6,151,834). coords(lb2gt_6,72,122,834).  visible(lb2gt_6,834). happensAt(inactive(lb2gt_6),834).
orientation(lb2gt_5,152,835). coords(lb2gt_5,105,77,835).  visible(lb2gt_5,835). happensAt(walking(lb2gt_5),835). orientation(lb2gt_6,151,835). coords(lb2gt_6,72,122,835).  visible(lb2gt_6,835). happensAt(inactive(lb2gt_6),835).
orientation(lb2gt_5,152,836). coords(lb2gt_5,105,78,836).  visible(lb2gt_5,836). happensAt(walking(lb2gt_5),836). orientation(lb2gt_6,151,836). coords(lb2gt_6,72,122,836).  visible(lb2gt_6,836). happensAt(inactive(lb2gt_6),836).
orientation(lb2gt_5,152,837). coords(lb2gt_5,104,79,837).  visible(lb2gt_5,837). happensAt(walking(lb2gt_5),837). orientation(lb2gt_6,151,837). coords(lb2gt_6,72,122,837).  visible(lb2gt_6,837). happensAt(inactive(lb2gt_6),837).
orientation(lb2gt_5,152,838). coords(lb2gt_5,102,79,838).  visible(lb2gt_5,838). happensAt(walking(lb2gt_5),838). orientation(lb2gt_6,151,838). coords(lb2gt_6,72,122,838).  visible(lb2gt_6,838). happensAt(inactive(lb2gt_6),838).
orientation(lb2gt_5,152,839). coords(lb2gt_5,102,80,839).  visible(lb2gt_5,839). happensAt(walking(lb2gt_5),839). orientation(lb2gt_6,151,839). coords(lb2gt_6,72,122,839).  visible(lb2gt_6,839). happensAt(inactive(lb2gt_6),839).
orientation(lb2gt_5,152,840). coords(lb2gt_5,100,81,840).  visible(lb2gt_5,840). happensAt(walking(lb2gt_5),840). orientation(lb2gt_6,151,840). coords(lb2gt_6,72,122,840).  visible(lb2gt_6,840). happensAt(inactive(lb2gt_6),840).
orientation(lb2gt_5,152,841). coords(lb2gt_5,98,81,841).  visible(lb2gt_5,841). happensAt(walking(lb2gt_5),841). orientation(lb2gt_6,151,841). coords(lb2gt_6,72,122,841).  visible(lb2gt_6,841). happensAt(inactive(lb2gt_6),841).
orientation(lb2gt_5,152,842). coords(lb2gt_5,97,82,842).  visible(lb2gt_5,842). happensAt(walking(lb2gt_5),842). orientation(lb2gt_6,151,842). coords(lb2gt_6,72,122,842).  visible(lb2gt_6,842). happensAt(inactive(lb2gt_6),842).
orientation(lb2gt_5,152,843). coords(lb2gt_5,97,82,843).  visible(lb2gt_5,843). happensAt(walking(lb2gt_5),843). orientation(lb2gt_6,151,843). coords(lb2gt_6,72,122,843).  visible(lb2gt_6,843). happensAt(inactive(lb2gt_6),843).
orientation(lb2gt_5,152,844). coords(lb2gt_5,96,83,844).  visible(lb2gt_5,844). happensAt(walking(lb2gt_5),844). orientation(lb2gt_6,151,844). coords(lb2gt_6,72,122,844).  visible(lb2gt_6,844). happensAt(inactive(lb2gt_6),844).
orientation(lb2gt_5,152,845). coords(lb2gt_5,96,83,845).  visible(lb2gt_5,845). happensAt(walking(lb2gt_5),845). orientation(lb2gt_6,151,845). coords(lb2gt_6,72,122,845).  visible(lb2gt_6,845). happensAt(inactive(lb2gt_6),845).
orientation(lb2gt_5,152,846). coords(lb2gt_5,96,85,846).  visible(lb2gt_5,846). happensAt(walking(lb2gt_5),846). orientation(lb2gt_6,151,846). coords(lb2gt_6,72,122,846).  visible(lb2gt_6,846). happensAt(inactive(lb2gt_6),846).
orientation(lb2gt_5,152,847). coords(lb2gt_5,95,86,847).  visible(lb2gt_5,847). happensAt(walking(lb2gt_5),847). orientation(lb2gt_6,151,847). coords(lb2gt_6,72,122,847).  visible(lb2gt_6,847). happensAt(inactive(lb2gt_6),847).
orientation(lb2gt_5,152,848). coords(lb2gt_5,95,87,848).  visible(lb2gt_5,848). happensAt(walking(lb2gt_5),848). orientation(lb2gt_6,151,848). coords(lb2gt_6,72,122,848).  visible(lb2gt_6,848). happensAt(inactive(lb2gt_6),848).
orientation(lb2gt_5,152,849). coords(lb2gt_5,94,89,849).  visible(lb2gt_5,849). happensAt(walking(lb2gt_5),849). orientation(lb2gt_6,151,849). coords(lb2gt_6,72,122,849).  visible(lb2gt_6,849). happensAt(inactive(lb2gt_6),849).
orientation(lb2gt_5,152,850). coords(lb2gt_5,93,89,850).  visible(lb2gt_5,850). happensAt(walking(lb2gt_5),850). orientation(lb2gt_6,151,850). coords(lb2gt_6,72,122,850).  visible(lb2gt_6,850). happensAt(inactive(lb2gt_6),850).
orientation(lb2gt_5,152,851). coords(lb2gt_5,92,90,851).  visible(lb2gt_5,851). happensAt(walking(lb2gt_5),851). orientation(lb2gt_6,151,851). coords(lb2gt_6,72,122,851).  visible(lb2gt_6,851). happensAt(inactive(lb2gt_6),851).
orientation(lb2gt_5,152,852). coords(lb2gt_5,91,90,852).  visible(lb2gt_5,852). happensAt(walking(lb2gt_5),852). orientation(lb2gt_6,151,852). coords(lb2gt_6,72,122,852).  visible(lb2gt_6,852). happensAt(inactive(lb2gt_6),852).
orientation(lb2gt_5,152,853). coords(lb2gt_5,90,90,853).  visible(lb2gt_5,853). happensAt(walking(lb2gt_5),853). orientation(lb2gt_6,151,853). coords(lb2gt_6,72,122,853).  visible(lb2gt_6,853). happensAt(inactive(lb2gt_6),853).
orientation(lb2gt_5,152,854). coords(lb2gt_5,88,91,854).  visible(lb2gt_5,854). happensAt(walking(lb2gt_5),854). orientation(lb2gt_6,151,854). coords(lb2gt_6,72,122,854).  visible(lb2gt_6,854). happensAt(inactive(lb2gt_6),854).
orientation(lb2gt_5,152,855). coords(lb2gt_5,87,91,855).  visible(lb2gt_5,855). happensAt(walking(lb2gt_5),855). orientation(lb2gt_6,151,855). coords(lb2gt_6,72,122,855).  visible(lb2gt_6,855). happensAt(inactive(lb2gt_6),855).
#show holdsAt/2.