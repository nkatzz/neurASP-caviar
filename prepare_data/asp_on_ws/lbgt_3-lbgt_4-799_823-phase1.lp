
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

	orientation(lbgt_3,61,799). coords(lbgt_3,146,162,799).  visible(lbgt_3,799). happensAt(inactive(lbgt_3),799). orientation(lbgt_4,139,799). coords(lbgt_4,82,96,799).  visible(lbgt_4,799). happensAt(walking(lbgt_4),799).
orientation(lbgt_3,61,800). coords(lbgt_3,146,162,800).  visible(lbgt_3,800). happensAt(inactive(lbgt_3),800). orientation(lbgt_4,139,800). coords(lbgt_4,82,96,800).  visible(lbgt_4,800). happensAt(walking(lbgt_4),800).
orientation(lbgt_3,61,801). coords(lbgt_3,146,162,801).  visible(lbgt_3,801). happensAt(inactive(lbgt_3),801). orientation(lbgt_4,139,801). coords(lbgt_4,82,96,801).  visible(lbgt_4,801). happensAt(walking(lbgt_4),801).
orientation(lbgt_3,61,802). coords(lbgt_3,146,162,802).  visible(lbgt_3,802). happensAt(inactive(lbgt_3),802). orientation(lbgt_4,139,802). coords(lbgt_4,82,95,802).  visible(lbgt_4,802). happensAt(walking(lbgt_4),802).
orientation(lbgt_3,61,803). coords(lbgt_3,146,162,803).  visible(lbgt_3,803). happensAt(inactive(lbgt_3),803). orientation(lbgt_4,139,803). coords(lbgt_4,82,95,803).  visible(lbgt_4,803). happensAt(walking(lbgt_4),803).
orientation(lbgt_3,61,804). coords(lbgt_3,146,162,804).  visible(lbgt_3,804). happensAt(inactive(lbgt_3),804). orientation(lbgt_4,139,804). coords(lbgt_4,82,95,804).  visible(lbgt_4,804). happensAt(walking(lbgt_4),804).
orientation(lbgt_3,61,805). coords(lbgt_3,146,162,805).  visible(lbgt_3,805). happensAt(inactive(lbgt_3),805). orientation(lbgt_4,139,805). coords(lbgt_4,82,94,805).  visible(lbgt_4,805). happensAt(walking(lbgt_4),805).
orientation(lbgt_3,61,806). coords(lbgt_3,146,162,806).  visible(lbgt_3,806). happensAt(inactive(lbgt_3),806). orientation(lbgt_4,139,806). coords(lbgt_4,82,93,806).  visible(lbgt_4,806). happensAt(walking(lbgt_4),806).
orientation(lbgt_3,61,807). coords(lbgt_3,146,162,807).  visible(lbgt_3,807). happensAt(inactive(lbgt_3),807). orientation(lbgt_4,139,807). coords(lbgt_4,81,92,807).  visible(lbgt_4,807). happensAt(walking(lbgt_4),807).
orientation(lbgt_3,61,808). coords(lbgt_3,146,162,808).  visible(lbgt_3,808). happensAt(inactive(lbgt_3),808). orientation(lbgt_4,144,808). coords(lbgt_4,81,91,808).  visible(lbgt_4,808). happensAt(walking(lbgt_4),808).
orientation(lbgt_3,61,809). coords(lbgt_3,146,162,809).  visible(lbgt_3,809). happensAt(inactive(lbgt_3),809). orientation(lbgt_4,144,809). coords(lbgt_4,81,90,809).  visible(lbgt_4,809). happensAt(walking(lbgt_4),809).
orientation(lbgt_3,61,810). coords(lbgt_3,146,162,810).  visible(lbgt_3,810). happensAt(inactive(lbgt_3),810). orientation(lbgt_4,144,810). coords(lbgt_4,81,90,810).  visible(lbgt_4,810). happensAt(walking(lbgt_4),810).
orientation(lbgt_3,61,811). coords(lbgt_3,146,162,811).  visible(lbgt_3,811). happensAt(inactive(lbgt_3),811). orientation(lbgt_4,144,811). coords(lbgt_4,81,89,811).  visible(lbgt_4,811). happensAt(walking(lbgt_4),811).
orientation(lbgt_3,61,812). coords(lbgt_3,146,162,812).  visible(lbgt_3,812). happensAt(inactive(lbgt_3),812). orientation(lbgt_4,144,812). coords(lbgt_4,81,89,812).  visible(lbgt_4,812). happensAt(walking(lbgt_4),812).
orientation(lbgt_3,61,813). coords(lbgt_3,146,162,813).  visible(lbgt_3,813). happensAt(inactive(lbgt_3),813). orientation(lbgt_4,144,813). coords(lbgt_4,81,89,813).  visible(lbgt_4,813). happensAt(walking(lbgt_4),813).
orientation(lbgt_3,61,814). coords(lbgt_3,146,162,814).  visible(lbgt_3,814). happensAt(inactive(lbgt_3),814). orientation(lbgt_4,144,814). coords(lbgt_4,81,89,814).  visible(lbgt_4,814). happensAt(walking(lbgt_4),814).
orientation(lbgt_3,61,815). coords(lbgt_3,146,162,815).  visible(lbgt_3,815). happensAt(inactive(lbgt_3),815). orientation(lbgt_4,144,815). coords(lbgt_4,82,89,815).  visible(lbgt_4,815). happensAt(walking(lbgt_4),815).
orientation(lbgt_3,61,816). coords(lbgt_3,146,162,816).  visible(lbgt_3,816). happensAt(inactive(lbgt_3),816). orientation(lbgt_4,136,816). coords(lbgt_4,82,88,816).  visible(lbgt_4,816). happensAt(walking(lbgt_4),816).
orientation(lbgt_3,61,817). coords(lbgt_3,146,162,817).  visible(lbgt_3,817). happensAt(inactive(lbgt_3),817). orientation(lbgt_4,136,817). coords(lbgt_4,82,87,817).  visible(lbgt_4,817). happensAt(walking(lbgt_4),817).
orientation(lbgt_3,61,818). coords(lbgt_3,146,162,818).  visible(lbgt_3,818). happensAt(inactive(lbgt_3),818). orientation(lbgt_4,136,818). coords(lbgt_4,82,87,818).  visible(lbgt_4,818). happensAt(walking(lbgt_4),818).
orientation(lbgt_3,61,819). coords(lbgt_3,146,162,819).  visible(lbgt_3,819). happensAt(inactive(lbgt_3),819). orientation(lbgt_4,136,819). coords(lbgt_4,82,87,819).  visible(lbgt_4,819). happensAt(walking(lbgt_4),819).
orientation(lbgt_3,61,820). coords(lbgt_3,146,162,820).  visible(lbgt_3,820). happensAt(inactive(lbgt_3),820). orientation(lbgt_4,136,820). coords(lbgt_4,82,85,820).  visible(lbgt_4,820). happensAt(walking(lbgt_4),820).
orientation(lbgt_3,61,821). coords(lbgt_3,146,162,821).  visible(lbgt_3,821). happensAt(inactive(lbgt_3),821). orientation(lbgt_4,136,821). coords(lbgt_4,82,82,821).  visible(lbgt_4,821). happensAt(walking(lbgt_4),821).
orientation(lbgt_3,61,822). coords(lbgt_3,146,162,822).  visible(lbgt_3,822). happensAt(inactive(lbgt_3),822). orientation(lbgt_4,136,822). coords(lbgt_4,82,82,822).  visible(lbgt_4,822). happensAt(walking(lbgt_4),822).
orientation(lbgt_3,61,823). coords(lbgt_3,146,162,823).  visible(lbgt_3,823). happensAt(inactive(lbgt_3),823). orientation(lbgt_4,136,823). coords(lbgt_4,82,82,823).  visible(lbgt_4,823). happensAt(walking(lbgt_4),823).
#show holdsAt/2.