
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

	orientation(lbpugt_2,97,779). coords(lbpugt_2,295,99,779).  visible(lbpugt_2,779). happensAt(active(lbpugt_2),779). orientation(lbpugt_4,135,779). coords(lbpugt_4,219,34,779).  visible(lbpugt_4,779). happensAt(inactive(lbpugt_4),779).
orientation(lbpugt_2,97,780). coords(lbpugt_2,295,99,780).  visible(lbpugt_2,780). happensAt(active(lbpugt_2),780). orientation(lbpugt_4,135,780). coords(lbpugt_4,219,34,780).  visible(lbpugt_4,780). happensAt(inactive(lbpugt_4),780).
orientation(lbpugt_2,97,781). coords(lbpugt_2,295,99,781).  visible(lbpugt_2,781). happensAt(active(lbpugt_2),781). orientation(lbpugt_4,135,781). coords(lbpugt_4,219,34,781).  visible(lbpugt_4,781). happensAt(inactive(lbpugt_4),781).
orientation(lbpugt_2,97,782). coords(lbpugt_2,295,99,782).  visible(lbpugt_2,782). happensAt(active(lbpugt_2),782). orientation(lbpugt_4,135,782). coords(lbpugt_4,219,34,782).  visible(lbpugt_4,782). happensAt(inactive(lbpugt_4),782).
orientation(lbpugt_2,97,783). coords(lbpugt_2,295,99,783).  visible(lbpugt_2,783). happensAt(active(lbpugt_2),783). orientation(lbpugt_4,135,783). coords(lbpugt_4,219,34,783).  visible(lbpugt_4,783). happensAt(inactive(lbpugt_4),783).
orientation(lbpugt_2,97,784). coords(lbpugt_2,295,99,784).  visible(lbpugt_2,784). happensAt(active(lbpugt_2),784). orientation(lbpugt_4,135,784). coords(lbpugt_4,219,34,784).  visible(lbpugt_4,784). happensAt(inactive(lbpugt_4),784).
orientation(lbpugt_2,97,785). coords(lbpugt_2,295,99,785).  visible(lbpugt_2,785). happensAt(active(lbpugt_2),785). orientation(lbpugt_4,135,785). coords(lbpugt_4,219,34,785).  visible(lbpugt_4,785). happensAt(inactive(lbpugt_4),785).
orientation(lbpugt_2,97,786). coords(lbpugt_2,295,99,786).  visible(lbpugt_2,786). happensAt(active(lbpugt_2),786). orientation(lbpugt_4,135,786). coords(lbpugt_4,219,34,786).  visible(lbpugt_4,786). happensAt(inactive(lbpugt_4),786).
orientation(lbpugt_2,97,787). coords(lbpugt_2,295,99,787).  visible(lbpugt_2,787). happensAt(active(lbpugt_2),787). orientation(lbpugt_4,135,787). coords(lbpugt_4,219,34,787).  visible(lbpugt_4,787). happensAt(inactive(lbpugt_4),787).
orientation(lbpugt_2,97,788). coords(lbpugt_2,295,99,788).  visible(lbpugt_2,788). happensAt(active(lbpugt_2),788). orientation(lbpugt_4,135,788). coords(lbpugt_4,219,34,788).  visible(lbpugt_4,788). happensAt(inactive(lbpugt_4),788).
orientation(lbpugt_2,97,789). coords(lbpugt_2,295,99,789).  visible(lbpugt_2,789). happensAt(active(lbpugt_2),789). orientation(lbpugt_4,135,789). coords(lbpugt_4,219,34,789).  visible(lbpugt_4,789). happensAt(inactive(lbpugt_4),789).
orientation(lbpugt_2,97,790). coords(lbpugt_2,295,99,790).  visible(lbpugt_2,790). happensAt(active(lbpugt_2),790). orientation(lbpugt_4,135,790). coords(lbpugt_4,219,34,790).  visible(lbpugt_4,790). happensAt(inactive(lbpugt_4),790).
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
#show holdsAt/2.