
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

	orientation(lb2gt_5,161,795). coords(lb2gt_5,142,44,795).  visible(lb2gt_5,795). happensAt(walking(lb2gt_5),795). orientation(lb2gt_6,151,795). coords(lb2gt_6,72,122,795).  visible(lb2gt_6,795). happensAt(inactive(lb2gt_6),795).
orientation(lb2gt_5,163,796). coords(lb2gt_5,141,46,796).  visible(lb2gt_5,796). happensAt(walking(lb2gt_5),796). orientation(lb2gt_6,151,796). coords(lb2gt_6,72,122,796).  visible(lb2gt_6,796). happensAt(inactive(lb2gt_6),796).
orientation(lb2gt_5,155,797). coords(lb2gt_5,139,46,797).  visible(lb2gt_5,797). happensAt(walking(lb2gt_5),797). orientation(lb2gt_6,151,797). coords(lb2gt_6,72,122,797).  visible(lb2gt_6,797). happensAt(inactive(lb2gt_6),797).
orientation(lb2gt_5,148,798). coords(lb2gt_5,139,47,798).  visible(lb2gt_5,798). happensAt(walking(lb2gt_5),798). orientation(lb2gt_6,151,798). coords(lb2gt_6,72,122,798).  visible(lb2gt_6,798). happensAt(inactive(lb2gt_6),798).
orientation(lb2gt_5,152,799). coords(lb2gt_5,137,47,799).  visible(lb2gt_5,799). happensAt(walking(lb2gt_5),799). orientation(lb2gt_6,151,799). coords(lb2gt_6,72,122,799).  visible(lb2gt_6,799). happensAt(inactive(lb2gt_6),799).
orientation(lb2gt_5,152,800). coords(lb2gt_5,135,47,800).  visible(lb2gt_5,800). happensAt(walking(lb2gt_5),800). orientation(lb2gt_6,151,800). coords(lb2gt_6,72,122,800).  visible(lb2gt_6,800). happensAt(inactive(lb2gt_6),800).
orientation(lb2gt_5,152,801). coords(lb2gt_5,133,48,801).  visible(lb2gt_5,801). happensAt(walking(lb2gt_5),801). orientation(lb2gt_6,151,801). coords(lb2gt_6,72,122,801).  visible(lb2gt_6,801). happensAt(inactive(lb2gt_6),801).
orientation(lb2gt_5,152,802). coords(lb2gt_5,131,48,802).  visible(lb2gt_5,802). happensAt(walking(lb2gt_5),802). orientation(lb2gt_6,151,802). coords(lb2gt_6,72,122,802).  visible(lb2gt_6,802). happensAt(inactive(lb2gt_6),802).
orientation(lb2gt_5,152,803). coords(lb2gt_5,131,49,803).  visible(lb2gt_5,803). happensAt(walking(lb2gt_5),803). orientation(lb2gt_6,151,803). coords(lb2gt_6,72,122,803).  visible(lb2gt_6,803). happensAt(inactive(lb2gt_6),803).
orientation(lb2gt_5,152,804). coords(lb2gt_5,130,49,804).  visible(lb2gt_5,804). happensAt(walking(lb2gt_5),804). orientation(lb2gt_6,151,804). coords(lb2gt_6,72,122,804).  visible(lb2gt_6,804). happensAt(inactive(lb2gt_6),804).
orientation(lb2gt_5,152,805). coords(lb2gt_5,129,50,805).  visible(lb2gt_5,805). happensAt(walking(lb2gt_5),805). orientation(lb2gt_6,151,805). coords(lb2gt_6,72,122,805).  visible(lb2gt_6,805). happensAt(inactive(lb2gt_6),805).
orientation(lb2gt_5,152,806). coords(lb2gt_5,129,51,806).  visible(lb2gt_5,806). happensAt(walking(lb2gt_5),806). orientation(lb2gt_6,151,806). coords(lb2gt_6,72,122,806).  visible(lb2gt_6,806). happensAt(inactive(lb2gt_6),806).
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
#show holdsAt/2.