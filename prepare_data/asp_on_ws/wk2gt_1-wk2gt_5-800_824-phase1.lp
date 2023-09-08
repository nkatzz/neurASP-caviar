
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

	orientation(wk2gt_1,125,800). coords(wk2gt_1,115,53,800).  visible(wk2gt_1,800). happensAt(inactive(wk2gt_1),800). orientation(wk2gt_5,161,800). coords(wk2gt_5,29,123,800).  happensAt(appear(wk2gt_5),800). happensAt(walking(wk2gt_5),800).
orientation(wk2gt_1,125,801). coords(wk2gt_1,115,53,801).  visible(wk2gt_1,801). happensAt(inactive(wk2gt_1),801). orientation(wk2gt_5,161,801). coords(wk2gt_5,29,123,801).  visible(wk2gt_5,801). happensAt(walking(wk2gt_5),801).
orientation(wk2gt_1,125,802). coords(wk2gt_1,115,53,802).  visible(wk2gt_1,802). happensAt(inactive(wk2gt_1),802). orientation(wk2gt_5,161,802). coords(wk2gt_5,29,123,802).  visible(wk2gt_5,802). happensAt(walking(wk2gt_5),802).
orientation(wk2gt_1,125,803). coords(wk2gt_1,115,53,803).  visible(wk2gt_1,803). happensAt(inactive(wk2gt_1),803). orientation(wk2gt_5,161,803). coords(wk2gt_5,29,123,803).  visible(wk2gt_5,803). happensAt(walking(wk2gt_5),803).
orientation(wk2gt_1,125,804). coords(wk2gt_1,115,53,804).  visible(wk2gt_1,804). happensAt(inactive(wk2gt_1),804). orientation(wk2gt_5,161,804). coords(wk2gt_5,29,121,804).  visible(wk2gt_5,804). happensAt(walking(wk2gt_5),804).
orientation(wk2gt_1,125,805). coords(wk2gt_1,115,53,805).  visible(wk2gt_1,805). happensAt(inactive(wk2gt_1),805). orientation(wk2gt_5,161,805). coords(wk2gt_5,29,121,805).  visible(wk2gt_5,805). happensAt(walking(wk2gt_5),805).
orientation(wk2gt_1,125,806). coords(wk2gt_1,115,53,806).  visible(wk2gt_1,806). happensAt(inactive(wk2gt_1),806). orientation(wk2gt_5,161,806). coords(wk2gt_5,29,120,806).  visible(wk2gt_5,806). happensAt(walking(wk2gt_5),806).
orientation(wk2gt_1,125,807). coords(wk2gt_1,115,53,807).  visible(wk2gt_1,807). happensAt(inactive(wk2gt_1),807). orientation(wk2gt_5,161,807). coords(wk2gt_5,29,120,807).  visible(wk2gt_5,807). happensAt(walking(wk2gt_5),807).
orientation(wk2gt_1,125,808). coords(wk2gt_1,115,53,808).  visible(wk2gt_1,808). happensAt(inactive(wk2gt_1),808). orientation(wk2gt_5,161,808). coords(wk2gt_5,30,119,808).  visible(wk2gt_5,808). happensAt(walking(wk2gt_5),808).
orientation(wk2gt_1,125,809). coords(wk2gt_1,115,53,809).  visible(wk2gt_1,809). happensAt(inactive(wk2gt_1),809). orientation(wk2gt_5,161,809). coords(wk2gt_5,30,119,809).  visible(wk2gt_5,809). happensAt(walking(wk2gt_5),809).
orientation(wk2gt_1,125,810). coords(wk2gt_1,115,53,810).  visible(wk2gt_1,810). happensAt(inactive(wk2gt_1),810). orientation(wk2gt_5,161,810). coords(wk2gt_5,30,118,810).  visible(wk2gt_5,810). happensAt(walking(wk2gt_5),810).
orientation(wk2gt_1,125,811). coords(wk2gt_1,115,53,811).  visible(wk2gt_1,811). happensAt(inactive(wk2gt_1),811). orientation(wk2gt_5,161,811). coords(wk2gt_5,31,118,811).  visible(wk2gt_5,811). happensAt(walking(wk2gt_5),811).
orientation(wk2gt_1,125,812). coords(wk2gt_1,115,53,812).  visible(wk2gt_1,812). happensAt(inactive(wk2gt_1),812). orientation(wk2gt_5,150,812). coords(wk2gt_5,31,117,812).  visible(wk2gt_5,812). happensAt(walking(wk2gt_5),812).
orientation(wk2gt_1,125,813). coords(wk2gt_1,115,53,813).  visible(wk2gt_1,813). happensAt(inactive(wk2gt_1),813). orientation(wk2gt_5,150,813). coords(wk2gt_5,31,117,813).  visible(wk2gt_5,813). happensAt(walking(wk2gt_5),813).
orientation(wk2gt_1,125,814). coords(wk2gt_1,115,53,814).  visible(wk2gt_1,814). happensAt(inactive(wk2gt_1),814). orientation(wk2gt_5,150,814). coords(wk2gt_5,31,116,814).  visible(wk2gt_5,814). happensAt(walking(wk2gt_5),814).
orientation(wk2gt_1,125,815). coords(wk2gt_1,115,53,815).  visible(wk2gt_1,815). happensAt(inactive(wk2gt_1),815). orientation(wk2gt_5,150,815). coords(wk2gt_5,31,116,815).  visible(wk2gt_5,815). happensAt(walking(wk2gt_5),815).
orientation(wk2gt_1,125,816). coords(wk2gt_1,115,53,816).  visible(wk2gt_1,816). happensAt(inactive(wk2gt_1),816). orientation(wk2gt_5,150,816). coords(wk2gt_5,31,116,816).  visible(wk2gt_5,816). happensAt(walking(wk2gt_5),816).
orientation(wk2gt_1,125,817). coords(wk2gt_1,115,53,817).  visible(wk2gt_1,817). happensAt(inactive(wk2gt_1),817). orientation(wk2gt_5,150,817). coords(wk2gt_5,31,115,817).  visible(wk2gt_5,817). happensAt(walking(wk2gt_5),817).
orientation(wk2gt_1,125,818). coords(wk2gt_1,115,53,818).  visible(wk2gt_1,818). happensAt(inactive(wk2gt_1),818). orientation(wk2gt_5,150,818). coords(wk2gt_5,31,115,818).  visible(wk2gt_5,818). happensAt(walking(wk2gt_5),818).
orientation(wk2gt_1,125,819). coords(wk2gt_1,115,53,819).  visible(wk2gt_1,819). happensAt(inactive(wk2gt_1),819). orientation(wk2gt_5,150,819). coords(wk2gt_5,31,115,819).  visible(wk2gt_5,819). happensAt(walking(wk2gt_5),819).
orientation(wk2gt_1,125,820). coords(wk2gt_1,115,53,820).  visible(wk2gt_1,820). happensAt(inactive(wk2gt_1),820). orientation(wk2gt_5,150,820). coords(wk2gt_5,31,115,820).  visible(wk2gt_5,820). happensAt(walking(wk2gt_5),820).
orientation(wk2gt_1,125,821). coords(wk2gt_1,115,53,821).  visible(wk2gt_1,821). happensAt(inactive(wk2gt_1),821). orientation(wk2gt_5,150,821). coords(wk2gt_5,32,114,821).  visible(wk2gt_5,821). happensAt(walking(wk2gt_5),821).
orientation(wk2gt_1,125,822). coords(wk2gt_1,115,53,822).  visible(wk2gt_1,822). happensAt(inactive(wk2gt_1),822). orientation(wk2gt_5,150,822). coords(wk2gt_5,32,114,822).  visible(wk2gt_5,822). happensAt(walking(wk2gt_5),822).
orientation(wk2gt_1,125,823). coords(wk2gt_1,115,53,823).  visible(wk2gt_1,823). happensAt(inactive(wk2gt_1),823). orientation(wk2gt_5,150,823). coords(wk2gt_5,32,113,823).  visible(wk2gt_5,823). happensAt(walking(wk2gt_5),823).
orientation(wk2gt_1,125,824). coords(wk2gt_1,115,53,824).  visible(wk2gt_1,824). happensAt(inactive(wk2gt_1),824). orientation(wk2gt_5,150,824). coords(wk2gt_5,32,112,824).  visible(wk2gt_5,824). happensAt(walking(wk2gt_5),824).
#show holdsAt/2.