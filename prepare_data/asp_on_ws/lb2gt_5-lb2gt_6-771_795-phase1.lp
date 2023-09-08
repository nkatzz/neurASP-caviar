
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

	orientation(lb2gt_5,132,771). coords(lb2gt_5,159,34,771).  visible(lb2gt_5,771). happensAt(walking(lb2gt_5),771). orientation(lb2gt_6,151,771). coords(lb2gt_6,72,122,771).  visible(lb2gt_6,771). happensAt(inactive(lb2gt_6),771).
orientation(lb2gt_5,132,772). coords(lb2gt_5,158,35,772).  visible(lb2gt_5,772). happensAt(walking(lb2gt_5),772). orientation(lb2gt_6,151,772). coords(lb2gt_6,72,122,772).  visible(lb2gt_6,772). happensAt(inactive(lb2gt_6),772).
orientation(lb2gt_5,132,773). coords(lb2gt_5,157,35,773).  visible(lb2gt_5,773). happensAt(walking(lb2gt_5),773). orientation(lb2gt_6,151,773). coords(lb2gt_6,72,122,773).  visible(lb2gt_6,773). happensAt(inactive(lb2gt_6),773).
orientation(lb2gt_5,132,774). coords(lb2gt_5,155,35,774).  visible(lb2gt_5,774). happensAt(walking(lb2gt_5),774). orientation(lb2gt_6,151,774). coords(lb2gt_6,72,122,774).  visible(lb2gt_6,774). happensAt(inactive(lb2gt_6),774).
orientation(lb2gt_5,132,775). coords(lb2gt_5,154,35,775).  visible(lb2gt_5,775). happensAt(walking(lb2gt_5),775). orientation(lb2gt_6,151,775). coords(lb2gt_6,72,122,775).  visible(lb2gt_6,775). happensAt(inactive(lb2gt_6),775).
orientation(lb2gt_5,132,776). coords(lb2gt_5,153,35,776).  visible(lb2gt_5,776). happensAt(walking(lb2gt_5),776). orientation(lb2gt_6,151,776). coords(lb2gt_6,72,122,776).  visible(lb2gt_6,776). happensAt(inactive(lb2gt_6),776).
orientation(lb2gt_5,132,777). coords(lb2gt_5,152,35,777).  visible(lb2gt_5,777). happensAt(walking(lb2gt_5),777). orientation(lb2gt_6,151,777). coords(lb2gt_6,72,122,777).  visible(lb2gt_6,777). happensAt(inactive(lb2gt_6),777).
orientation(lb2gt_5,132,778). coords(lb2gt_5,152,36,778).  visible(lb2gt_5,778). happensAt(walking(lb2gt_5),778). orientation(lb2gt_6,151,778). coords(lb2gt_6,72,122,778).  visible(lb2gt_6,778). happensAt(inactive(lb2gt_6),778).
orientation(lb2gt_5,132,779). coords(lb2gt_5,152,36,779).  visible(lb2gt_5,779). happensAt(walking(lb2gt_5),779). orientation(lb2gt_6,151,779). coords(lb2gt_6,72,122,779).  visible(lb2gt_6,779). happensAt(inactive(lb2gt_6),779).
orientation(lb2gt_5,132,780). coords(lb2gt_5,152,38,780).  visible(lb2gt_5,780). happensAt(walking(lb2gt_5),780). orientation(lb2gt_6,151,780). coords(lb2gt_6,72,122,780).  visible(lb2gt_6,780). happensAt(inactive(lb2gt_6),780).
orientation(lb2gt_5,132,781). coords(lb2gt_5,152,39,781).  visible(lb2gt_5,781). happensAt(walking(lb2gt_5),781). orientation(lb2gt_6,151,781). coords(lb2gt_6,72,122,781).  visible(lb2gt_6,781). happensAt(inactive(lb2gt_6),781).
orientation(lb2gt_5,132,782). coords(lb2gt_5,152,39,782).  visible(lb2gt_5,782). happensAt(walking(lb2gt_5),782). orientation(lb2gt_6,151,782). coords(lb2gt_6,72,122,782).  visible(lb2gt_6,782). happensAt(inactive(lb2gt_6),782).
orientation(lb2gt_5,132,783). coords(lb2gt_5,151,40,783).  visible(lb2gt_5,783). happensAt(walking(lb2gt_5),783). orientation(lb2gt_6,151,783). coords(lb2gt_6,72,122,783).  visible(lb2gt_6,783). happensAt(inactive(lb2gt_6),783).
orientation(lb2gt_5,136,784). coords(lb2gt_5,151,40,784).  visible(lb2gt_5,784). happensAt(walking(lb2gt_5),784). orientation(lb2gt_6,151,784). coords(lb2gt_6,72,122,784).  visible(lb2gt_6,784). happensAt(inactive(lb2gt_6),784).
orientation(lb2gt_5,144,785). coords(lb2gt_5,151,41,785).  visible(lb2gt_5,785). happensAt(walking(lb2gt_5),785). orientation(lb2gt_6,151,785). coords(lb2gt_6,72,122,785).  visible(lb2gt_6,785). happensAt(inactive(lb2gt_6),785).
orientation(lb2gt_5,147,786). coords(lb2gt_5,150,41,786).  visible(lb2gt_5,786). happensAt(walking(lb2gt_5),786). orientation(lb2gt_6,151,786). coords(lb2gt_6,72,122,786).  visible(lb2gt_6,786). happensAt(inactive(lb2gt_6),786).
orientation(lb2gt_5,151,787). coords(lb2gt_5,149,41,787).  visible(lb2gt_5,787). happensAt(walking(lb2gt_5),787). orientation(lb2gt_6,151,787). coords(lb2gt_6,72,122,787).  visible(lb2gt_6,787). happensAt(inactive(lb2gt_6),787).
orientation(lb2gt_5,156,788). coords(lb2gt_5,149,42,788).  visible(lb2gt_5,788). happensAt(walking(lb2gt_5),788). orientation(lb2gt_6,151,788). coords(lb2gt_6,72,122,788).  visible(lb2gt_6,788). happensAt(inactive(lb2gt_6),788).
orientation(lb2gt_5,157,789). coords(lb2gt_5,147,42,789).  visible(lb2gt_5,789). happensAt(walking(lb2gt_5),789). orientation(lb2gt_6,151,789). coords(lb2gt_6,72,122,789).  visible(lb2gt_6,789). happensAt(inactive(lb2gt_6),789).
orientation(lb2gt_5,156,790). coords(lb2gt_5,145,42,790).  visible(lb2gt_5,790). happensAt(walking(lb2gt_5),790). orientation(lb2gt_6,151,790). coords(lb2gt_6,72,122,790).  visible(lb2gt_6,790). happensAt(inactive(lb2gt_6),790).
orientation(lb2gt_5,159,791). coords(lb2gt_5,145,42,791).  visible(lb2gt_5,791). happensAt(walking(lb2gt_5),791). orientation(lb2gt_6,151,791). coords(lb2gt_6,72,122,791).  visible(lb2gt_6,791). happensAt(inactive(lb2gt_6),791).
orientation(lb2gt_5,155,792). coords(lb2gt_5,144,43,792).  visible(lb2gt_5,792). happensAt(walking(lb2gt_5),792). orientation(lb2gt_6,151,792). coords(lb2gt_6,72,122,792).  visible(lb2gt_6,792). happensAt(inactive(lb2gt_6),792).
orientation(lb2gt_5,162,793). coords(lb2gt_5,143,43,793).  visible(lb2gt_5,793). happensAt(walking(lb2gt_5),793). orientation(lb2gt_6,151,793). coords(lb2gt_6,72,122,793).  visible(lb2gt_6,793). happensAt(inactive(lb2gt_6),793).
orientation(lb2gt_5,162,794). coords(lb2gt_5,142,44,794).  visible(lb2gt_5,794). happensAt(walking(lb2gt_5),794). orientation(lb2gt_6,151,794). coords(lb2gt_6,72,122,794).  visible(lb2gt_6,794). happensAt(inactive(lb2gt_6),794).
orientation(lb2gt_5,161,795). coords(lb2gt_5,142,44,795).  visible(lb2gt_5,795). happensAt(walking(lb2gt_5),795). orientation(lb2gt_6,151,795). coords(lb2gt_6,72,122,795).  visible(lb2gt_6,795). happensAt(inactive(lb2gt_6),795).
#show holdsAt/2.