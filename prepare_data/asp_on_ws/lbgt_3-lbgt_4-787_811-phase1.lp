
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

	orientation(lbgt_3,61,787). coords(lbgt_3,146,162,787).  visible(lbgt_3,787). happensAt(inactive(lbgt_3),787). orientation(lbgt_4,151,787). coords(lbgt_4,82,107,787).  visible(lbgt_4,787). happensAt(walking(lbgt_4),787).
orientation(lbgt_3,61,788). coords(lbgt_3,146,162,788).  visible(lbgt_3,788). happensAt(inactive(lbgt_3),788). orientation(lbgt_4,151,788). coords(lbgt_4,82,106,788).  visible(lbgt_4,788). happensAt(walking(lbgt_4),788).
orientation(lbgt_3,61,789). coords(lbgt_3,146,162,789).  visible(lbgt_3,789). happensAt(inactive(lbgt_3),789). orientation(lbgt_4,145,789). coords(lbgt_4,82,106,789).  visible(lbgt_4,789). happensAt(walking(lbgt_4),789).
orientation(lbgt_3,61,790). coords(lbgt_3,146,162,790).  visible(lbgt_3,790). happensAt(inactive(lbgt_3),790). orientation(lbgt_4,145,790). coords(lbgt_4,82,105,790).  visible(lbgt_4,790). happensAt(walking(lbgt_4),790).
orientation(lbgt_3,61,791). coords(lbgt_3,146,162,791).  visible(lbgt_3,791). happensAt(inactive(lbgt_3),791). orientation(lbgt_4,141,791). coords(lbgt_4,82,105,791).  visible(lbgt_4,791). happensAt(walking(lbgt_4),791).
orientation(lbgt_3,61,792). coords(lbgt_3,146,162,792).  visible(lbgt_3,792). happensAt(inactive(lbgt_3),792). orientation(lbgt_4,139,792). coords(lbgt_4,82,105,792).  visible(lbgt_4,792). happensAt(walking(lbgt_4),792).
orientation(lbgt_3,61,793). coords(lbgt_3,146,162,793).  visible(lbgt_3,793). happensAt(inactive(lbgt_3),793). orientation(lbgt_4,139,793). coords(lbgt_4,82,104,793).  visible(lbgt_4,793). happensAt(walking(lbgt_4),793).
orientation(lbgt_3,61,794). coords(lbgt_3,146,162,794).  visible(lbgt_3,794). happensAt(inactive(lbgt_3),794). orientation(lbgt_4,139,794). coords(lbgt_4,82,103,794).  visible(lbgt_4,794). happensAt(walking(lbgt_4),794).
orientation(lbgt_3,61,795). coords(lbgt_3,146,162,795).  visible(lbgt_3,795). happensAt(inactive(lbgt_3),795). orientation(lbgt_4,139,795). coords(lbgt_4,82,101,795).  visible(lbgt_4,795). happensAt(walking(lbgt_4),795).
orientation(lbgt_3,61,796). coords(lbgt_3,146,162,796).  visible(lbgt_3,796). happensAt(inactive(lbgt_3),796). orientation(lbgt_4,139,796). coords(lbgt_4,82,100,796).  visible(lbgt_4,796). happensAt(walking(lbgt_4),796).
orientation(lbgt_3,61,797). coords(lbgt_3,146,162,797).  visible(lbgt_3,797). happensAt(inactive(lbgt_3),797). orientation(lbgt_4,139,797). coords(lbgt_4,82,98,797).  visible(lbgt_4,797). happensAt(walking(lbgt_4),797).
orientation(lbgt_3,61,798). coords(lbgt_3,146,162,798).  visible(lbgt_3,798). happensAt(inactive(lbgt_3),798). orientation(lbgt_4,139,798). coords(lbgt_4,82,97,798).  visible(lbgt_4,798). happensAt(walking(lbgt_4),798).
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
#show holdsAt/2.