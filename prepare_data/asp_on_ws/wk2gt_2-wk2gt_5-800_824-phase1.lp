
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

	orientation(wk2gt_2,142,800). coords(wk2gt_2,99,63,800).  visible(wk2gt_2,800). happensAt(inactive(wk2gt_2),800). orientation(wk2gt_5,161,800). coords(wk2gt_5,29,123,800).  happensAt(appear(wk2gt_5),800). happensAt(walking(wk2gt_5),800).
orientation(wk2gt_2,142,801). coords(wk2gt_2,99,63,801).  visible(wk2gt_2,801). happensAt(inactive(wk2gt_2),801). orientation(wk2gt_5,161,801). coords(wk2gt_5,29,123,801).  visible(wk2gt_5,801). happensAt(walking(wk2gt_5),801).
orientation(wk2gt_2,142,802). coords(wk2gt_2,99,63,802).  visible(wk2gt_2,802). happensAt(inactive(wk2gt_2),802). orientation(wk2gt_5,161,802). coords(wk2gt_5,29,123,802).  visible(wk2gt_5,802). happensAt(walking(wk2gt_5),802).
orientation(wk2gt_2,142,803). coords(wk2gt_2,99,63,803).  visible(wk2gt_2,803). happensAt(inactive(wk2gt_2),803). orientation(wk2gt_5,161,803). coords(wk2gt_5,29,123,803).  visible(wk2gt_5,803). happensAt(walking(wk2gt_5),803).
orientation(wk2gt_2,142,804). coords(wk2gt_2,99,63,804).  visible(wk2gt_2,804). happensAt(inactive(wk2gt_2),804). orientation(wk2gt_5,161,804). coords(wk2gt_5,29,121,804).  visible(wk2gt_5,804). happensAt(walking(wk2gt_5),804).
orientation(wk2gt_2,142,805). coords(wk2gt_2,99,63,805).  visible(wk2gt_2,805). happensAt(inactive(wk2gt_2),805). orientation(wk2gt_5,161,805). coords(wk2gt_5,29,121,805).  visible(wk2gt_5,805). happensAt(walking(wk2gt_5),805).
orientation(wk2gt_2,142,806). coords(wk2gt_2,99,63,806).  visible(wk2gt_2,806). happensAt(inactive(wk2gt_2),806). orientation(wk2gt_5,161,806). coords(wk2gt_5,29,120,806).  visible(wk2gt_5,806). happensAt(walking(wk2gt_5),806).
orientation(wk2gt_2,142,807). coords(wk2gt_2,99,63,807).  visible(wk2gt_2,807). happensAt(inactive(wk2gt_2),807). orientation(wk2gt_5,161,807). coords(wk2gt_5,29,120,807).  visible(wk2gt_5,807). happensAt(walking(wk2gt_5),807).
orientation(wk2gt_2,142,808). coords(wk2gt_2,99,63,808).  visible(wk2gt_2,808). happensAt(inactive(wk2gt_2),808). orientation(wk2gt_5,161,808). coords(wk2gt_5,30,119,808).  visible(wk2gt_5,808). happensAt(walking(wk2gt_5),808).
orientation(wk2gt_2,142,809). coords(wk2gt_2,99,63,809).  visible(wk2gt_2,809). happensAt(inactive(wk2gt_2),809). orientation(wk2gt_5,161,809). coords(wk2gt_5,30,119,809).  visible(wk2gt_5,809). happensAt(walking(wk2gt_5),809).
orientation(wk2gt_2,142,810). coords(wk2gt_2,99,63,810).  visible(wk2gt_2,810). happensAt(inactive(wk2gt_2),810). orientation(wk2gt_5,161,810). coords(wk2gt_5,30,118,810).  visible(wk2gt_5,810). happensAt(walking(wk2gt_5),810).
orientation(wk2gt_2,142,811). coords(wk2gt_2,99,63,811).  visible(wk2gt_2,811). happensAt(inactive(wk2gt_2),811). orientation(wk2gt_5,161,811). coords(wk2gt_5,31,118,811).  visible(wk2gt_5,811). happensAt(walking(wk2gt_5),811).
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
#show holdsAt/2.