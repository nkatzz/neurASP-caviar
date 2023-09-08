
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

	orientation(ms3ggt_3,162,786). coords(ms3ggt_3,54,97,786).  visible(ms3ggt_3,786). happensAt(walking(ms3ggt_3),786). orientation(ms3ggt_4,120,786). coords(ms3ggt_4,144,26,786).  visible(ms3ggt_4,786). happensAt(walking(ms3ggt_4),786).
orientation(ms3ggt_3,162,787). coords(ms3ggt_3,54,98,787).  visible(ms3ggt_3,787). happensAt(walking(ms3ggt_3),787). orientation(ms3ggt_4,120,787). coords(ms3ggt_4,143,26,787).  visible(ms3ggt_4,787). happensAt(walking(ms3ggt_4),787).
orientation(ms3ggt_3,162,788). coords(ms3ggt_3,54,98,788).  visible(ms3ggt_3,788). happensAt(walking(ms3ggt_3),788). orientation(ms3ggt_4,120,788). coords(ms3ggt_4,143,26,788).  visible(ms3ggt_4,788). happensAt(walking(ms3ggt_4),788).
orientation(ms3ggt_3,162,789). coords(ms3ggt_3,54,99,789).  visible(ms3ggt_3,789). happensAt(walking(ms3ggt_3),789). orientation(ms3ggt_4,120,789). coords(ms3ggt_4,143,26,789).  visible(ms3ggt_4,789). happensAt(walking(ms3ggt_4),789).
orientation(ms3ggt_3,162,790). coords(ms3ggt_3,53,99,790).  visible(ms3ggt_3,790). happensAt(walking(ms3ggt_3),790). orientation(ms3ggt_4,120,790). coords(ms3ggt_4,143,26,790).  visible(ms3ggt_4,790). happensAt(walking(ms3ggt_4),790).
orientation(ms3ggt_3,162,791). coords(ms3ggt_3,53,100,791).  visible(ms3ggt_3,791). happensAt(walking(ms3ggt_3),791). orientation(ms3ggt_4,120,791). coords(ms3ggt_4,143,26,791).  visible(ms3ggt_4,791). happensAt(walking(ms3ggt_4),791).
orientation(ms3ggt_3,168,792). coords(ms3ggt_3,52,100,792).  visible(ms3ggt_3,792). happensAt(walking(ms3ggt_3),792). orientation(ms3ggt_4,120,792). coords(ms3ggt_4,143,26,792).  visible(ms3ggt_4,792). happensAt(walking(ms3ggt_4),792).
orientation(ms3ggt_3,168,793). coords(ms3ggt_3,52,101,793).  visible(ms3ggt_3,793). happensAt(walking(ms3ggt_3),793). orientation(ms3ggt_4,120,793). coords(ms3ggt_4,143,26,793).  visible(ms3ggt_4,793). happensAt(walking(ms3ggt_4),793).
orientation(ms3ggt_3,150,794). coords(ms3ggt_3,52,102,794).  visible(ms3ggt_3,794). happensAt(walking(ms3ggt_3),794). orientation(ms3ggt_4,120,794). coords(ms3ggt_4,143,26,794).  visible(ms3ggt_4,794). happensAt(walking(ms3ggt_4),794).
orientation(ms3ggt_3,157,795). coords(ms3ggt_3,47,103,795).  visible(ms3ggt_3,795). happensAt(walking(ms3ggt_3),795). orientation(ms3ggt_4,120,795). coords(ms3ggt_4,142,27,795).  visible(ms3ggt_4,795). happensAt(walking(ms3ggt_4),795).
orientation(ms3ggt_3,157,796). coords(ms3ggt_3,45,105,796).  visible(ms3ggt_3,796). happensAt(walking(ms3ggt_3),796). orientation(ms3ggt_4,120,796). coords(ms3ggt_4,142,27,796).  visible(ms3ggt_4,796). happensAt(walking(ms3ggt_4),796).
orientation(ms3ggt_3,157,797). coords(ms3ggt_3,45,105,797).  visible(ms3ggt_3,797). happensAt(walking(ms3ggt_3),797). orientation(ms3ggt_4,120,797). coords(ms3ggt_4,142,27,797).  visible(ms3ggt_4,797). happensAt(walking(ms3ggt_4),797).
orientation(ms3ggt_3,157,798). coords(ms3ggt_3,45,107,798).  visible(ms3ggt_3,798). happensAt(walking(ms3ggt_3),798). orientation(ms3ggt_4,120,798). coords(ms3ggt_4,142,28,798).  visible(ms3ggt_4,798). happensAt(walking(ms3ggt_4),798).
orientation(ms3ggt_3,157,799). coords(ms3ggt_3,45,108,799).  visible(ms3ggt_3,799). happensAt(walking(ms3ggt_3),799). orientation(ms3ggt_4,120,799). coords(ms3ggt_4,142,28,799).  visible(ms3ggt_4,799). happensAt(walking(ms3ggt_4),799).
orientation(ms3ggt_3,157,800). coords(ms3ggt_3,45,108,800).  visible(ms3ggt_3,800). happensAt(walking(ms3ggt_3),800). orientation(ms3ggt_4,120,800). coords(ms3ggt_4,142,28,800).  visible(ms3ggt_4,800). happensAt(walking(ms3ggt_4),800).
orientation(ms3ggt_3,147,801). coords(ms3ggt_3,44,110,801).  visible(ms3ggt_3,801). happensAt(walking(ms3ggt_3),801). orientation(ms3ggt_4,120,801). coords(ms3ggt_4,141,29,801).  visible(ms3ggt_4,801). happensAt(walking(ms3ggt_4),801).
orientation(ms3ggt_3,147,802). coords(ms3ggt_3,44,112,802).  visible(ms3ggt_3,802). happensAt(walking(ms3ggt_3),802). orientation(ms3ggt_4,120,802). coords(ms3ggt_4,140,29,802).  visible(ms3ggt_4,802). happensAt(walking(ms3ggt_4),802).
orientation(ms3ggt_3,147,803). coords(ms3ggt_3,44,113,803).  visible(ms3ggt_3,803). happensAt(walking(ms3ggt_3),803). orientation(ms3ggt_4,120,803). coords(ms3ggt_4,139,29,803).  visible(ms3ggt_4,803). happensAt(walking(ms3ggt_4),803).
orientation(ms3ggt_3,147,804). coords(ms3ggt_3,46,115,804).  visible(ms3ggt_3,804). happensAt(walking(ms3ggt_3),804). orientation(ms3ggt_4,120,804). coords(ms3ggt_4,139,29,804).  visible(ms3ggt_4,804). happensAt(walking(ms3ggt_4),804).
orientation(ms3ggt_3,147,805). coords(ms3ggt_3,46,115,805).  visible(ms3ggt_3,805). happensAt(walking(ms3ggt_3),805). orientation(ms3ggt_4,120,805). coords(ms3ggt_4,139,29,805).  visible(ms3ggt_4,805). happensAt(walking(ms3ggt_4),805).
orientation(ms3ggt_3,147,806). coords(ms3ggt_3,46,117,806).  visible(ms3ggt_3,806). happensAt(walking(ms3ggt_3),806). orientation(ms3ggt_4,120,806). coords(ms3ggt_4,139,29,806).  visible(ms3ggt_4,806). happensAt(walking(ms3ggt_4),806).
orientation(ms3ggt_3,147,807). coords(ms3ggt_3,44,119,807).  visible(ms3ggt_3,807). happensAt(walking(ms3ggt_3),807). orientation(ms3ggt_4,120,807). coords(ms3ggt_4,139,30,807).  visible(ms3ggt_4,807). happensAt(walking(ms3ggt_4),807).
orientation(ms3ggt_3,147,808). coords(ms3ggt_3,44,119,808).  visible(ms3ggt_3,808). happensAt(walking(ms3ggt_3),808). orientation(ms3ggt_4,120,808). coords(ms3ggt_4,139,30,808).  visible(ms3ggt_4,808). happensAt(walking(ms3ggt_4),808).
orientation(ms3ggt_3,147,809). coords(ms3ggt_3,45,119,809).  visible(ms3ggt_3,809). happensAt(walking(ms3ggt_3),809). orientation(ms3ggt_4,120,809). coords(ms3ggt_4,139,30,809).  visible(ms3ggt_4,809). happensAt(walking(ms3ggt_4),809).
orientation(ms3ggt_3,147,810). coords(ms3ggt_3,44,121,810).  visible(ms3ggt_3,810). happensAt(walking(ms3ggt_3),810). orientation(ms3ggt_4,120,810). coords(ms3ggt_4,138,30,810).  visible(ms3ggt_4,810). happensAt(walking(ms3ggt_4),810).
#show holdsAt/2.