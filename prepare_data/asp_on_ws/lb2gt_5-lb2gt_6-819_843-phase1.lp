
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

	orientation(lb2gt_5,152,819). coords(lb2gt_5,118,61,819).  visible(lb2gt_5,819). happensAt(walking(lb2gt_5),819). orientation(lb2gt_6,151,819). coords(lb2gt_6,72,122,819).  visible(lb2gt_6,819). happensAt(inactive(lb2gt_6),819).
orientation(lb2gt_5,152,820). coords(lb2gt_5,117,62,820).  visible(lb2gt_5,820). happensAt(walking(lb2gt_5),820). orientation(lb2gt_6,151,820). coords(lb2gt_6,72,122,820).  visible(lb2gt_6,820). happensAt(inactive(lb2gt_6),820).
orientation(lb2gt_5,152,821). coords(lb2gt_5,117,63,821).  visible(lb2gt_5,821). happensAt(walking(lb2gt_5),821). orientation(lb2gt_6,151,821). coords(lb2gt_6,72,122,821).  visible(lb2gt_6,821). happensAt(inactive(lb2gt_6),821).
orientation(lb2gt_5,152,822). coords(lb2gt_5,117,64,822).  visible(lb2gt_5,822). happensAt(walking(lb2gt_5),822). orientation(lb2gt_6,151,822). coords(lb2gt_6,72,122,822).  visible(lb2gt_6,822). happensAt(inactive(lb2gt_6),822).
orientation(lb2gt_5,152,823). coords(lb2gt_5,116,65,823).  visible(lb2gt_5,823). happensAt(walking(lb2gt_5),823). orientation(lb2gt_6,151,823). coords(lb2gt_6,72,122,823).  visible(lb2gt_6,823). happensAt(inactive(lb2gt_6),823).
orientation(lb2gt_5,152,824). coords(lb2gt_5,116,65,824).  visible(lb2gt_5,824). happensAt(walking(lb2gt_5),824). orientation(lb2gt_6,151,824). coords(lb2gt_6,72,122,824).  visible(lb2gt_6,824). happensAt(inactive(lb2gt_6),824).
orientation(lb2gt_5,152,825). coords(lb2gt_5,114,66,825).  visible(lb2gt_5,825). happensAt(walking(lb2gt_5),825). orientation(lb2gt_6,151,825). coords(lb2gt_6,72,122,825).  visible(lb2gt_6,825). happensAt(inactive(lb2gt_6),825).
orientation(lb2gt_5,152,826). coords(lb2gt_5,112,66,826).  visible(lb2gt_5,826). happensAt(walking(lb2gt_5),826). orientation(lb2gt_6,151,826). coords(lb2gt_6,72,122,826).  visible(lb2gt_6,826). happensAt(inactive(lb2gt_6),826).
orientation(lb2gt_5,152,827). coords(lb2gt_5,110,66,827).  visible(lb2gt_5,827). happensAt(walking(lb2gt_5),827). orientation(lb2gt_6,151,827). coords(lb2gt_6,72,122,827).  visible(lb2gt_6,827). happensAt(inactive(lb2gt_6),827).
orientation(lb2gt_5,152,828). coords(lb2gt_5,110,67,828).  visible(lb2gt_5,828). happensAt(walking(lb2gt_5),828). orientation(lb2gt_6,151,828). coords(lb2gt_6,72,122,828).  visible(lb2gt_6,828). happensAt(inactive(lb2gt_6),828).
orientation(lb2gt_5,152,829). coords(lb2gt_5,108,68,829).  visible(lb2gt_5,829). happensAt(walking(lb2gt_5),829). orientation(lb2gt_6,151,829). coords(lb2gt_6,72,122,829).  visible(lb2gt_6,829). happensAt(inactive(lb2gt_6),829).
orientation(lb2gt_5,152,830). coords(lb2gt_5,108,70,830).  visible(lb2gt_5,830). happensAt(walking(lb2gt_5),830). orientation(lb2gt_6,151,830). coords(lb2gt_6,72,122,830).  visible(lb2gt_6,830). happensAt(inactive(lb2gt_6),830).
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
#show holdsAt/2.