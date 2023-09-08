
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

	orientation(br1gt_2,12,790). coords(br1gt_2,86,209,790).  visible(br1gt_2,790). happensAt(inactive(br1gt_2),790). orientation(br1gt_3,99,790). coords(br1gt_3,299,163,790).  visible(br1gt_3,790). happensAt(walking(br1gt_3),790).
orientation(br1gt_2,12,791). coords(br1gt_2,86,209,791).  visible(br1gt_2,791). happensAt(inactive(br1gt_2),791). orientation(br1gt_3,99,791). coords(br1gt_3,298,165,791).  visible(br1gt_3,791). happensAt(walking(br1gt_3),791).
orientation(br1gt_2,12,792). coords(br1gt_2,86,209,792).  visible(br1gt_2,792). happensAt(inactive(br1gt_2),792). orientation(br1gt_3,99,792). coords(br1gt_3,298,166,792).  visible(br1gt_3,792). happensAt(walking(br1gt_3),792).
orientation(br1gt_2,12,793). coords(br1gt_2,86,209,793).  visible(br1gt_2,793). happensAt(inactive(br1gt_2),793). orientation(br1gt_3,97,793). coords(br1gt_3,297,167,793).  visible(br1gt_3,793). happensAt(walking(br1gt_3),793).
orientation(br1gt_2,12,794). coords(br1gt_2,86,209,794).  visible(br1gt_2,794). happensAt(inactive(br1gt_2),794). orientation(br1gt_3,97,794). coords(br1gt_3,297,168,794).  visible(br1gt_3,794). happensAt(walking(br1gt_3),794).
orientation(br1gt_2,12,795). coords(br1gt_2,86,209,795).  visible(br1gt_2,795). happensAt(inactive(br1gt_2),795). orientation(br1gt_3,97,795). coords(br1gt_3,296,169,795).  visible(br1gt_3,795). happensAt(walking(br1gt_3),795).
orientation(br1gt_2,12,796). coords(br1gt_2,86,209,796).  visible(br1gt_2,796). happensAt(inactive(br1gt_2),796). orientation(br1gt_3,97,796). coords(br1gt_3,296,170,796).  visible(br1gt_3,796). happensAt(walking(br1gt_3),796).
orientation(br1gt_2,12,797). coords(br1gt_2,86,209,797).  visible(br1gt_2,797). happensAt(inactive(br1gt_2),797). orientation(br1gt_3,112,797). coords(br1gt_3,296,175,797).  visible(br1gt_3,797). happensAt(walking(br1gt_3),797).
orientation(br1gt_2,12,798). coords(br1gt_2,86,209,798).  visible(br1gt_2,798). happensAt(inactive(br1gt_2),798). orientation(br1gt_3,112,798). coords(br1gt_3,296,179,798).  visible(br1gt_3,798). happensAt(walking(br1gt_3),798).
orientation(br1gt_2,12,799). coords(br1gt_2,86,209,799).  visible(br1gt_2,799). happensAt(inactive(br1gt_2),799). orientation(br1gt_3,112,799). coords(br1gt_3,294,183,799).  visible(br1gt_3,799). happensAt(walking(br1gt_3),799).
orientation(br1gt_2,12,800). coords(br1gt_2,86,209,800).  visible(br1gt_2,800). happensAt(inactive(br1gt_2),800). orientation(br1gt_3,112,800). coords(br1gt_3,294,185,800).  visible(br1gt_3,800). happensAt(walking(br1gt_3),800).
orientation(br1gt_2,12,801). coords(br1gt_2,86,209,801).  visible(br1gt_2,801). happensAt(inactive(br1gt_2),801). orientation(br1gt_3,108,801). coords(br1gt_3,294,188,801).  visible(br1gt_3,801). happensAt(walking(br1gt_3),801).
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
#show holdsAt/2.