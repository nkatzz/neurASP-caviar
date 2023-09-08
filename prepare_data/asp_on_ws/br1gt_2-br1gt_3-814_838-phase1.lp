
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

	orientation(br1gt_2,12,814). coords(br1gt_2,86,209,814).  visible(br1gt_2,814). happensAt(inactive(br1gt_2),814). orientation(br1gt_3,88,814). coords(br1gt_3,289,205,814).  visible(br1gt_3,814). happensAt(walking(br1gt_3),814).
orientation(br1gt_2,12,815). coords(br1gt_2,86,209,815).  visible(br1gt_2,815). happensAt(inactive(br1gt_2),815). orientation(br1gt_3,88,815). coords(br1gt_3,289,207,815).  visible(br1gt_3,815). happensAt(walking(br1gt_3),815).
orientation(br1gt_2,12,816). coords(br1gt_2,86,209,816).  visible(br1gt_2,816). happensAt(inactive(br1gt_2),816). orientation(br1gt_3,88,816). coords(br1gt_3,287,207,816).  visible(br1gt_3,816). happensAt(walking(br1gt_3),816).
orientation(br1gt_2,12,817). coords(br1gt_2,86,209,817).  visible(br1gt_2,817). happensAt(inactive(br1gt_2),817). orientation(br1gt_3,88,817). coords(br1gt_3,288,209,817).  visible(br1gt_3,817). happensAt(walking(br1gt_3),817).
orientation(br1gt_2,12,818). coords(br1gt_2,86,209,818).  visible(br1gt_2,818). happensAt(inactive(br1gt_2),818). orientation(br1gt_3,88,818). coords(br1gt_3,286,210,818).  visible(br1gt_3,818). happensAt(walking(br1gt_3),818).
orientation(br1gt_2,12,819). coords(br1gt_2,86,209,819).  visible(br1gt_2,819). happensAt(inactive(br1gt_2),819). orientation(br1gt_3,88,819). coords(br1gt_3,286,211,819).  visible(br1gt_3,819). happensAt(walking(br1gt_3),819).
orientation(br1gt_2,12,820). coords(br1gt_2,86,209,820).  visible(br1gt_2,820). happensAt(inactive(br1gt_2),820). orientation(br1gt_3,88,820). coords(br1gt_3,285,211,820).  visible(br1gt_3,820). happensAt(walking(br1gt_3),820).
orientation(br1gt_2,12,821). coords(br1gt_2,86,209,821).  visible(br1gt_2,821). happensAt(inactive(br1gt_2),821). orientation(br1gt_3,88,821). coords(br1gt_3,285,212,821).  visible(br1gt_3,821). happensAt(walking(br1gt_3),821).
orientation(br1gt_2,12,822). coords(br1gt_2,86,209,822).  visible(br1gt_2,822). happensAt(inactive(br1gt_2),822). orientation(br1gt_3,88,822). coords(br1gt_3,284,211,822).  visible(br1gt_3,822). happensAt(walking(br1gt_3),822).
orientation(br1gt_2,12,823). coords(br1gt_2,86,209,823).  visible(br1gt_2,823). happensAt(inactive(br1gt_2),823). orientation(br1gt_3,88,823). coords(br1gt_3,283,211,823).  visible(br1gt_3,823). happensAt(walking(br1gt_3),823).
orientation(br1gt_2,12,824). coords(br1gt_2,86,209,824).  visible(br1gt_2,824). happensAt(inactive(br1gt_2),824). orientation(br1gt_3,116,824). coords(br1gt_3,282,211,824).  visible(br1gt_3,824). happensAt(walking(br1gt_3),824).
orientation(br1gt_2,12,825). coords(br1gt_2,86,209,825).  visible(br1gt_2,825). happensAt(inactive(br1gt_2),825). orientation(br1gt_3,116,825). coords(br1gt_3,282,211,825).  visible(br1gt_3,825). happensAt(walking(br1gt_3),825).
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
#show holdsAt/2.