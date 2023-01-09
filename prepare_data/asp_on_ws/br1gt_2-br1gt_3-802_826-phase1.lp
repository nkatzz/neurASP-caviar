
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

	orientation(br1gt_2,12,802). coords(br1gt_2,86,209,802).  visible(br1gt_2,802). happensAt(inactive(br1gt_2),802). orientation(br1gt_3,108,802). coords(br1gt_3,294,190,802).  visible(br1gt_3,802). happensAt(walking(br1gt_3),802).
orientation(br1gt_2,12,803). coords(br1gt_2,86,209,803).  visible(br1gt_2,803). happensAt(inactive(br1gt_2),803). orientation(br1gt_3,108,803). coords(br1gt_3,294,190,803).  visible(br1gt_3,803). happensAt(walking(br1gt_3),803).
orientation(br1gt_2,12,804). coords(br1gt_2,86,209,804).  visible(br1gt_2,804). happensAt(inactive(br1gt_2),804). orientation(br1gt_3,108,804). coords(br1gt_3,294,193,804).  visible(br1gt_3,804). happensAt(walking(br1gt_3),804).
orientation(br1gt_2,12,805). coords(br1gt_2,86,209,805).  visible(br1gt_2,805). happensAt(inactive(br1gt_2),805). orientation(br1gt_3,108,805). coords(br1gt_3,294,193,805).  visible(br1gt_3,805). happensAt(walking(br1gt_3),805).
orientation(br1gt_2,12,806). coords(br1gt_2,86,209,806).  visible(br1gt_2,806). happensAt(inactive(br1gt_2),806). orientation(br1gt_3,108,806). coords(br1gt_3,294,193,806).  visible(br1gt_3,806). happensAt(walking(br1gt_3),806).
orientation(br1gt_2,12,807). coords(br1gt_2,86,209,807).  visible(br1gt_2,807). happensAt(inactive(br1gt_2),807). orientation(br1gt_3,100,807). coords(br1gt_3,294,195,807).  visible(br1gt_3,807). happensAt(walking(br1gt_3),807).
orientation(br1gt_2,12,808). coords(br1gt_2,86,209,808).  visible(br1gt_2,808). happensAt(inactive(br1gt_2),808). orientation(br1gt_3,96,808). coords(br1gt_3,293,196,808).  visible(br1gt_3,808). happensAt(walking(br1gt_3),808).
orientation(br1gt_2,12,809). coords(br1gt_2,86,209,809).  visible(br1gt_2,809). happensAt(inactive(br1gt_2),809). orientation(br1gt_3,96,809). coords(br1gt_3,293,198,809).  visible(br1gt_3,809). happensAt(walking(br1gt_3),809).
orientation(br1gt_2,12,810). coords(br1gt_2,86,209,810).  visible(br1gt_2,810). happensAt(inactive(br1gt_2),810). orientation(br1gt_3,96,810). coords(br1gt_3,292,198,810).  visible(br1gt_3,810). happensAt(walking(br1gt_3),810).
orientation(br1gt_2,12,811). coords(br1gt_2,86,209,811).  visible(br1gt_2,811). happensAt(inactive(br1gt_2),811). orientation(br1gt_3,96,811). coords(br1gt_3,291,199,811).  visible(br1gt_3,811). happensAt(walking(br1gt_3),811).
orientation(br1gt_2,12,812). coords(br1gt_2,86,209,812).  visible(br1gt_2,812). happensAt(inactive(br1gt_2),812). orientation(br1gt_3,93,812). coords(br1gt_3,290,201,812).  visible(br1gt_3,812). happensAt(walking(br1gt_3),812).
orientation(br1gt_2,12,813). coords(br1gt_2,86,209,813).  visible(br1gt_2,813). happensAt(inactive(br1gt_2),813). orientation(br1gt_3,93,813). coords(br1gt_3,290,204,813).  visible(br1gt_3,813). happensAt(walking(br1gt_3),813).
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
#show holdsAt/2.