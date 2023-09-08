
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

	orientation(wk2gt_2,142,812). coords(wk2gt_2,99,63,812).  visible(wk2gt_2,812). happensAt(inactive(wk2gt_2),812). orientation(wk2gt_5,150,812). coords(wk2gt_5,31,117,812).  visible(wk2gt_5,812). happensAt(walking(wk2gt_5),812).
orientation(wk2gt_2,142,813). coords(wk2gt_2,99,63,813).  visible(wk2gt_2,813). happensAt(inactive(wk2gt_2),813). orientation(wk2gt_5,150,813). coords(wk2gt_5,31,117,813).  visible(wk2gt_5,813). happensAt(walking(wk2gt_5),813).
orientation(wk2gt_2,142,814). coords(wk2gt_2,99,63,814).  visible(wk2gt_2,814). happensAt(inactive(wk2gt_2),814). orientation(wk2gt_5,150,814). coords(wk2gt_5,31,116,814).  visible(wk2gt_5,814). happensAt(walking(wk2gt_5),814).
orientation(wk2gt_2,142,815). coords(wk2gt_2,99,63,815).  visible(wk2gt_2,815). happensAt(inactive(wk2gt_2),815). orientation(wk2gt_5,150,815). coords(wk2gt_5,31,116,815).  visible(wk2gt_5,815). happensAt(walking(wk2gt_5),815).
orientation(wk2gt_2,142,816). coords(wk2gt_2,99,63,816).  visible(wk2gt_2,816). happensAt(inactive(wk2gt_2),816). orientation(wk2gt_5,150,816). coords(wk2gt_5,31,116,816).  visible(wk2gt_5,816). happensAt(walking(wk2gt_5),816).
orientation(wk2gt_2,142,817). coords(wk2gt_2,99,63,817).  visible(wk2gt_2,817). happensAt(inactive(wk2gt_2),817). orientation(wk2gt_5,150,817). coords(wk2gt_5,31,115,817).  visible(wk2gt_5,817). happensAt(walking(wk2gt_5),817).
orientation(wk2gt_2,142,818). coords(wk2gt_2,99,63,818).  visible(wk2gt_2,818). happensAt(inactive(wk2gt_2),818). orientation(wk2gt_5,150,818). coords(wk2gt_5,31,115,818).  visible(wk2gt_5,818). happensAt(walking(wk2gt_5),818).
orientation(wk2gt_2,142,819). coords(wk2gt_2,99,63,819).  visible(wk2gt_2,819). happensAt(inactive(wk2gt_2),819). orientation(wk2gt_5,150,819). coords(wk2gt_5,31,115,819).  visible(wk2gt_5,819). happensAt(walking(wk2gt_5),819).
orientation(wk2gt_2,142,820). coords(wk2gt_2,99,63,820).  visible(wk2gt_2,820). happensAt(inactive(wk2gt_2),820). orientation(wk2gt_5,150,820). coords(wk2gt_5,31,115,820).  visible(wk2gt_5,820). happensAt(walking(wk2gt_5),820).
orientation(wk2gt_2,142,821). coords(wk2gt_2,99,63,821).  visible(wk2gt_2,821). happensAt(inactive(wk2gt_2),821). orientation(wk2gt_5,150,821). coords(wk2gt_5,32,114,821).  visible(wk2gt_5,821). happensAt(walking(wk2gt_5),821).
orientation(wk2gt_2,142,822). coords(wk2gt_2,99,63,822).  visible(wk2gt_2,822). happensAt(inactive(wk2gt_2),822). orientation(wk2gt_5,150,822). coords(wk2gt_5,32,114,822).  visible(wk2gt_5,822). happensAt(walking(wk2gt_5),822).
orientation(wk2gt_2,142,823). coords(wk2gt_2,99,63,823).  visible(wk2gt_2,823). happensAt(inactive(wk2gt_2),823). orientation(wk2gt_5,150,823). coords(wk2gt_5,32,113,823).  visible(wk2gt_5,823). happensAt(walking(wk2gt_5),823).
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
#show holdsAt/2.