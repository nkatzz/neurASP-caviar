
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

	orientation(lbpugt_2,102,791). coords(lbpugt_2,294,99,791).  visible(lbpugt_2,791). happensAt(active(lbpugt_2),791). orientation(lbpugt_4,135,791). coords(lbpugt_4,219,34,791).  visible(lbpugt_4,791). happensAt(inactive(lbpugt_4),791).
orientation(lbpugt_2,102,792). coords(lbpugt_2,294,99,792).  visible(lbpugt_2,792). happensAt(active(lbpugt_2),792). orientation(lbpugt_4,135,792). coords(lbpugt_4,219,34,792).  visible(lbpugt_4,792). happensAt(inactive(lbpugt_4),792).
orientation(lbpugt_2,102,793). coords(lbpugt_2,294,99,793).  visible(lbpugt_2,793). happensAt(active(lbpugt_2),793). orientation(lbpugt_4,135,793). coords(lbpugt_4,219,34,793).  visible(lbpugt_4,793). happensAt(inactive(lbpugt_4),793).
orientation(lbpugt_2,102,794). coords(lbpugt_2,294,99,794).  visible(lbpugt_2,794). happensAt(active(lbpugt_2),794). orientation(lbpugt_4,135,794). coords(lbpugt_4,219,34,794).  visible(lbpugt_4,794). happensAt(inactive(lbpugt_4),794).
orientation(lbpugt_2,102,795). coords(lbpugt_2,294,99,795).  visible(lbpugt_2,795). happensAt(active(lbpugt_2),795). orientation(lbpugt_4,135,795). coords(lbpugt_4,219,34,795).  visible(lbpugt_4,795). happensAt(inactive(lbpugt_4),795).
orientation(lbpugt_2,102,796). coords(lbpugt_2,293,99,796).  visible(lbpugt_2,796). happensAt(active(lbpugt_2),796). orientation(lbpugt_4,135,796). coords(lbpugt_4,219,34,796).  visible(lbpugt_4,796). happensAt(inactive(lbpugt_4),796).
orientation(lbpugt_2,102,797). coords(lbpugt_2,293,99,797).  visible(lbpugt_2,797). happensAt(active(lbpugt_2),797). orientation(lbpugt_4,135,797). coords(lbpugt_4,219,34,797).  visible(lbpugt_4,797). happensAt(inactive(lbpugt_4),797).
orientation(lbpugt_2,108,798). coords(lbpugt_2,293,99,798).  visible(lbpugt_2,798). happensAt(active(lbpugt_2),798). orientation(lbpugt_4,135,798). coords(lbpugt_4,219,34,798).  visible(lbpugt_4,798). happensAt(inactive(lbpugt_4),798).
orientation(lbpugt_2,109,799). coords(lbpugt_2,292,99,799).  visible(lbpugt_2,799). happensAt(active(lbpugt_2),799). orientation(lbpugt_4,135,799). coords(lbpugt_4,219,34,799).  visible(lbpugt_4,799). happensAt(inactive(lbpugt_4),799).
orientation(lbpugt_2,105,800). coords(lbpugt_2,292,99,800).  visible(lbpugt_2,800). happensAt(active(lbpugt_2),800). orientation(lbpugt_4,135,800). coords(lbpugt_4,219,34,800).  visible(lbpugt_4,800). happensAt(inactive(lbpugt_4),800).
orientation(lbpugt_2,105,801). coords(lbpugt_2,292,99,801).  visible(lbpugt_2,801). happensAt(active(lbpugt_2),801). orientation(lbpugt_4,135,801). coords(lbpugt_4,219,34,801).  visible(lbpugt_4,801). happensAt(inactive(lbpugt_4),801).
orientation(lbpugt_2,105,802). coords(lbpugt_2,292,99,802).  visible(lbpugt_2,802). happensAt(active(lbpugt_2),802). orientation(lbpugt_4,135,802). coords(lbpugt_4,219,34,802).  visible(lbpugt_4,802). happensAt(inactive(lbpugt_4),802).
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
#show holdsAt/2.