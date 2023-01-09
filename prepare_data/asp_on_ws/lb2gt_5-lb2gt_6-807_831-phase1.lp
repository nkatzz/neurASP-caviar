
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

	orientation(lb2gt_5,152,807). coords(lb2gt_5,128,52,807).  visible(lb2gt_5,807). happensAt(walking(lb2gt_5),807). orientation(lb2gt_6,151,807). coords(lb2gt_6,72,122,807).  visible(lb2gt_6,807). happensAt(inactive(lb2gt_6),807).
orientation(lb2gt_5,152,808). coords(lb2gt_5,126,53,808).  visible(lb2gt_5,808). happensAt(walking(lb2gt_5),808). orientation(lb2gt_6,151,808). coords(lb2gt_6,72,122,808).  visible(lb2gt_6,808). happensAt(inactive(lb2gt_6),808).
orientation(lb2gt_5,152,809). coords(lb2gt_5,125,53,809).  visible(lb2gt_5,809). happensAt(walking(lb2gt_5),809). orientation(lb2gt_6,151,809). coords(lb2gt_6,72,122,809).  visible(lb2gt_6,809). happensAt(inactive(lb2gt_6),809).
orientation(lb2gt_5,152,810). coords(lb2gt_5,125,54,810).  visible(lb2gt_5,810). happensAt(walking(lb2gt_5),810). orientation(lb2gt_6,151,810). coords(lb2gt_6,72,122,810).  visible(lb2gt_6,810). happensAt(inactive(lb2gt_6),810).
orientation(lb2gt_5,152,811). coords(lb2gt_5,123,54,811).  visible(lb2gt_5,811). happensAt(walking(lb2gt_5),811). orientation(lb2gt_6,151,811). coords(lb2gt_6,72,122,811).  visible(lb2gt_6,811). happensAt(inactive(lb2gt_6),811).
orientation(lb2gt_5,152,812). coords(lb2gt_5,122,54,812).  visible(lb2gt_5,812). happensAt(walking(lb2gt_5),812). orientation(lb2gt_6,151,812). coords(lb2gt_6,72,122,812).  visible(lb2gt_6,812). happensAt(inactive(lb2gt_6),812).
orientation(lb2gt_5,152,813). coords(lb2gt_5,121,54,813).  visible(lb2gt_5,813). happensAt(walking(lb2gt_5),813). orientation(lb2gt_6,151,813). coords(lb2gt_6,72,122,813).  visible(lb2gt_6,813). happensAt(inactive(lb2gt_6),813).
orientation(lb2gt_5,152,814). coords(lb2gt_5,120,54,814).  visible(lb2gt_5,814). happensAt(walking(lb2gt_5),814). orientation(lb2gt_6,151,814). coords(lb2gt_6,72,122,814).  visible(lb2gt_6,814). happensAt(inactive(lb2gt_6),814).
orientation(lb2gt_5,152,815). coords(lb2gt_5,120,55,815).  visible(lb2gt_5,815). happensAt(walking(lb2gt_5),815). orientation(lb2gt_6,151,815). coords(lb2gt_6,72,122,815).  visible(lb2gt_6,815). happensAt(inactive(lb2gt_6),815).
orientation(lb2gt_5,152,816). coords(lb2gt_5,119,56,816).  visible(lb2gt_5,816). happensAt(walking(lb2gt_5),816). orientation(lb2gt_6,151,816). coords(lb2gt_6,72,122,816).  visible(lb2gt_6,816). happensAt(inactive(lb2gt_6),816).
orientation(lb2gt_5,152,817). coords(lb2gt_5,119,58,817).  visible(lb2gt_5,817). happensAt(walking(lb2gt_5),817). orientation(lb2gt_6,151,817). coords(lb2gt_6,72,122,817).  visible(lb2gt_6,817). happensAt(inactive(lb2gt_6),817).
orientation(lb2gt_5,152,818). coords(lb2gt_5,119,60,818).  visible(lb2gt_5,818). happensAt(walking(lb2gt_5),818). orientation(lb2gt_6,151,818). coords(lb2gt_6,72,122,818).  visible(lb2gt_6,818). happensAt(inactive(lb2gt_6),818).
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
#show holdsAt/2.