
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

	orientation(lbpugt_2,105,803). coords(lbpugt_2,292,100,803).  visible(lbpugt_2,803). happensAt(active(lbpugt_2),803). orientation(lbpugt_4,135,803). coords(lbpugt_4,219,34,803).  visible(lbpugt_4,803). happensAt(inactive(lbpugt_4),803).
orientation(lbpugt_2,105,804). coords(lbpugt_2,292,100,804).  visible(lbpugt_2,804). happensAt(active(lbpugt_2),804). orientation(lbpugt_4,135,804). coords(lbpugt_4,219,34,804).  visible(lbpugt_4,804). happensAt(inactive(lbpugt_4),804).
orientation(lbpugt_2,105,805). coords(lbpugt_2,292,100,805).  visible(lbpugt_2,805). happensAt(active(lbpugt_2),805). orientation(lbpugt_4,135,805). coords(lbpugt_4,219,34,805).  visible(lbpugt_4,805). happensAt(inactive(lbpugt_4),805).
orientation(lbpugt_2,103,806). coords(lbpugt_2,292,100,806).  visible(lbpugt_2,806). happensAt(active(lbpugt_2),806). orientation(lbpugt_4,135,806). coords(lbpugt_4,219,34,806).  visible(lbpugt_4,806). happensAt(inactive(lbpugt_4),806).
orientation(lbpugt_2,100,807). coords(lbpugt_2,292,100,807).  visible(lbpugt_2,807). happensAt(active(lbpugt_2),807). orientation(lbpugt_4,135,807). coords(lbpugt_4,219,34,807).  visible(lbpugt_4,807). happensAt(inactive(lbpugt_4),807).
orientation(lbpugt_2,100,808). coords(lbpugt_2,293,100,808).  visible(lbpugt_2,808). happensAt(active(lbpugt_2),808). orientation(lbpugt_4,135,808). coords(lbpugt_4,219,34,808).  visible(lbpugt_4,808). happensAt(inactive(lbpugt_4),808).
orientation(lbpugt_2,97,809). coords(lbpugt_2,293,100,809).  visible(lbpugt_2,809). happensAt(active(lbpugt_2),809). orientation(lbpugt_4,135,809). coords(lbpugt_4,219,34,809).  visible(lbpugt_4,809). happensAt(inactive(lbpugt_4),809).
orientation(lbpugt_2,97,810). coords(lbpugt_2,293,101,810).  visible(lbpugt_2,810). happensAt(active(lbpugt_2),810). orientation(lbpugt_4,135,810). coords(lbpugt_4,219,34,810).  visible(lbpugt_4,810). happensAt(inactive(lbpugt_4),810).
orientation(lbpugt_2,96,811). coords(lbpugt_2,293,101,811).  visible(lbpugt_2,811). happensAt(active(lbpugt_2),811). orientation(lbpugt_4,135,811). coords(lbpugt_4,219,34,811).  visible(lbpugt_4,811). happensAt(inactive(lbpugt_4),811).
orientation(lbpugt_2,96,812). coords(lbpugt_2,294,101,812).  visible(lbpugt_2,812). happensAt(active(lbpugt_2),812). orientation(lbpugt_4,135,812). coords(lbpugt_4,219,34,812).  visible(lbpugt_4,812). happensAt(inactive(lbpugt_4),812).
orientation(lbpugt_2,96,813). coords(lbpugt_2,294,101,813).  visible(lbpugt_2,813). happensAt(active(lbpugt_2),813). orientation(lbpugt_4,135,813). coords(lbpugt_4,219,34,813).  visible(lbpugt_4,813). happensAt(inactive(lbpugt_4),813).
orientation(lbpugt_2,91,814). coords(lbpugt_2,294,100,814).  visible(lbpugt_2,814). happensAt(active(lbpugt_2),814). orientation(lbpugt_4,135,814). coords(lbpugt_4,219,34,814).  visible(lbpugt_4,814). happensAt(inactive(lbpugt_4),814).
orientation(lbpugt_2,91,815). coords(lbpugt_2,294,102,815).  visible(lbpugt_2,815). happensAt(active(lbpugt_2),815). orientation(lbpugt_4,135,815). coords(lbpugt_4,219,34,815).  visible(lbpugt_4,815). happensAt(inactive(lbpugt_4),815).
orientation(lbpugt_2,91,816). coords(lbpugt_2,294,101,816).  visible(lbpugt_2,816). happensAt(active(lbpugt_2),816). orientation(lbpugt_4,135,816). coords(lbpugt_4,219,34,816).  visible(lbpugt_4,816). happensAt(inactive(lbpugt_4),816).
orientation(lbpugt_2,89,817). coords(lbpugt_2,294,101,817).  visible(lbpugt_2,817). happensAt(active(lbpugt_2),817). orientation(lbpugt_4,135,817). coords(lbpugt_4,219,34,817).  visible(lbpugt_4,817). happensAt(inactive(lbpugt_4),817).
orientation(lbpugt_2,89,818). coords(lbpugt_2,294,100,818).  visible(lbpugt_2,818). happensAt(active(lbpugt_2),818). orientation(lbpugt_4,135,818). coords(lbpugt_4,219,34,818).  visible(lbpugt_4,818). happensAt(inactive(lbpugt_4),818).
orientation(lbpugt_2,89,819). coords(lbpugt_2,294,100,819).  visible(lbpugt_2,819). happensAt(active(lbpugt_2),819). orientation(lbpugt_4,135,819). coords(lbpugt_4,219,34,819).  visible(lbpugt_4,819). happensAt(inactive(lbpugt_4),819).
orientation(lbpugt_2,89,820). coords(lbpugt_2,293,100,820).  visible(lbpugt_2,820). happensAt(active(lbpugt_2),820). orientation(lbpugt_4,135,820). coords(lbpugt_4,219,34,820).  visible(lbpugt_4,820). happensAt(inactive(lbpugt_4),820).
orientation(lbpugt_2,89,821). coords(lbpugt_2,293,100,821).  visible(lbpugt_2,821). happensAt(active(lbpugt_2),821). orientation(lbpugt_4,135,821). coords(lbpugt_4,219,34,821).  visible(lbpugt_4,821). happensAt(inactive(lbpugt_4),821).
orientation(lbpugt_2,89,822). coords(lbpugt_2,292,100,822).  visible(lbpugt_2,822). happensAt(active(lbpugt_2),822). orientation(lbpugt_4,135,822). coords(lbpugt_4,219,34,822).  visible(lbpugt_4,822). happensAt(inactive(lbpugt_4),822).
orientation(lbpugt_2,89,823). coords(lbpugt_2,292,100,823).  visible(lbpugt_2,823). happensAt(active(lbpugt_2),823). orientation(lbpugt_4,135,823). coords(lbpugt_4,219,34,823).  visible(lbpugt_4,823). happensAt(inactive(lbpugt_4),823).
orientation(lbpugt_2,89,824). coords(lbpugt_2,292,100,824).  visible(lbpugt_2,824). happensAt(active(lbpugt_2),824). orientation(lbpugt_4,135,824). coords(lbpugt_4,219,34,824).  visible(lbpugt_4,824). happensAt(inactive(lbpugt_4),824).
orientation(lbpugt_2,89,825). coords(lbpugt_2,292,100,825).  visible(lbpugt_2,825). happensAt(active(lbpugt_2),825). orientation(lbpugt_4,135,825). coords(lbpugt_4,219,34,825).  visible(lbpugt_4,825). happensAt(inactive(lbpugt_4),825).
orientation(lbpugt_2,89,826). coords(lbpugt_2,292,100,826).  visible(lbpugt_2,826). happensAt(active(lbpugt_2),826). orientation(lbpugt_4,135,826). coords(lbpugt_4,219,34,826).  visible(lbpugt_4,826). happensAt(inactive(lbpugt_4),826).
orientation(lbpugt_2,86,827). coords(lbpugt_2,292,100,827).  visible(lbpugt_2,827). happensAt(active(lbpugt_2),827). orientation(lbpugt_4,135,827). coords(lbpugt_4,219,34,827).  visible(lbpugt_4,827). happensAt(inactive(lbpugt_4),827).
#show holdsAt/2.