
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

	orientation(br1gt_2,12,766). coords(br1gt_2,86,209,766).  visible(br1gt_2,766). happensAt(inactive(br1gt_2),766). orientation(br1gt_3,107,766). coords(br1gt_3,301,115,766).  visible(br1gt_3,766). happensAt(walking(br1gt_3),766).
orientation(br1gt_2,12,767). coords(br1gt_2,86,209,767).  visible(br1gt_2,767). happensAt(inactive(br1gt_2),767). orientation(br1gt_3,107,767). coords(br1gt_3,300,121,767).  visible(br1gt_3,767). happensAt(walking(br1gt_3),767).
orientation(br1gt_2,12,768). coords(br1gt_2,86,209,768).  visible(br1gt_2,768). happensAt(inactive(br1gt_2),768). orientation(br1gt_3,107,768). coords(br1gt_3,300,123,768).  visible(br1gt_3,768). happensAt(walking(br1gt_3),768).
orientation(br1gt_2,12,769). coords(br1gt_2,86,209,769).  visible(br1gt_2,769). happensAt(inactive(br1gt_2),769). orientation(br1gt_3,107,769). coords(br1gt_3,300,125,769).  visible(br1gt_3,769). happensAt(walking(br1gt_3),769).
orientation(br1gt_2,12,770). coords(br1gt_2,86,209,770).  visible(br1gt_2,770). happensAt(inactive(br1gt_2),770). orientation(br1gt_3,107,770). coords(br1gt_3,300,127,770).  visible(br1gt_3,770). happensAt(walking(br1gt_3),770).
orientation(br1gt_2,12,771). coords(br1gt_2,86,209,771).  visible(br1gt_2,771). happensAt(inactive(br1gt_2),771). orientation(br1gt_3,107,771). coords(br1gt_3,300,128,771).  visible(br1gt_3,771). happensAt(walking(br1gt_3),771).
orientation(br1gt_2,12,772). coords(br1gt_2,86,209,772).  visible(br1gt_2,772). happensAt(inactive(br1gt_2),772). orientation(br1gt_3,107,772). coords(br1gt_3,300,130,772).  visible(br1gt_3,772). happensAt(walking(br1gt_3),772).
orientation(br1gt_2,12,773). coords(br1gt_2,86,209,773).  visible(br1gt_2,773). happensAt(inactive(br1gt_2),773). orientation(br1gt_3,107,773). coords(br1gt_3,300,131,773).  visible(br1gt_3,773). happensAt(walking(br1gt_3),773).
orientation(br1gt_2,12,774). coords(br1gt_2,86,209,774).  visible(br1gt_2,774). happensAt(inactive(br1gt_2),774). orientation(br1gt_3,107,774). coords(br1gt_3,301,132,774).  visible(br1gt_3,774). happensAt(walking(br1gt_3),774).
orientation(br1gt_2,12,775). coords(br1gt_2,86,209,775).  visible(br1gt_2,775). happensAt(inactive(br1gt_2),775). orientation(br1gt_3,107,775). coords(br1gt_3,301,133,775).  visible(br1gt_3,775). happensAt(walking(br1gt_3),775).
orientation(br1gt_2,12,776). coords(br1gt_2,86,209,776).  visible(br1gt_2,776). happensAt(inactive(br1gt_2),776). orientation(br1gt_3,107,776). coords(br1gt_3,302,134,776).  visible(br1gt_3,776). happensAt(walking(br1gt_3),776).
orientation(br1gt_2,12,777). coords(br1gt_2,86,209,777).  visible(br1gt_2,777). happensAt(inactive(br1gt_2),777). orientation(br1gt_3,107,777). coords(br1gt_3,302,135,777).  visible(br1gt_3,777). happensAt(walking(br1gt_3),777).
orientation(br1gt_2,12,778). coords(br1gt_2,86,209,778).  visible(br1gt_2,778). happensAt(inactive(br1gt_2),778). orientation(br1gt_3,107,778). coords(br1gt_3,302,136,778).  visible(br1gt_3,778). happensAt(walking(br1gt_3),778).
orientation(br1gt_2,12,779). coords(br1gt_2,86,209,779).  visible(br1gt_2,779). happensAt(inactive(br1gt_2),779). orientation(br1gt_3,107,779). coords(br1gt_3,302,137,779).  visible(br1gt_3,779). happensAt(walking(br1gt_3),779).
orientation(br1gt_2,12,780). coords(br1gt_2,86,209,780).  visible(br1gt_2,780). happensAt(inactive(br1gt_2),780). orientation(br1gt_3,107,780). coords(br1gt_3,301,138,780).  visible(br1gt_3,780). happensAt(walking(br1gt_3),780).
orientation(br1gt_2,12,781). coords(br1gt_2,86,209,781).  visible(br1gt_2,781). happensAt(inactive(br1gt_2),781). orientation(br1gt_3,107,781). coords(br1gt_3,301,139,781).  visible(br1gt_3,781). happensAt(walking(br1gt_3),781).
orientation(br1gt_2,12,782). coords(br1gt_2,86,209,782).  visible(br1gt_2,782). happensAt(inactive(br1gt_2),782). orientation(br1gt_3,107,782). coords(br1gt_3,301,143,782).  visible(br1gt_3,782). happensAt(walking(br1gt_3),782).
orientation(br1gt_2,12,783). coords(br1gt_2,86,209,783).  visible(br1gt_2,783). happensAt(inactive(br1gt_2),783). orientation(br1gt_3,106,783). coords(br1gt_3,301,148,783).  visible(br1gt_3,783). happensAt(walking(br1gt_3),783).
orientation(br1gt_2,12,784). coords(br1gt_2,86,209,784).  visible(br1gt_2,784). happensAt(inactive(br1gt_2),784). orientation(br1gt_3,102,784). coords(br1gt_3,301,151,784).  visible(br1gt_3,784). happensAt(walking(br1gt_3),784).
orientation(br1gt_2,12,785). coords(br1gt_2,86,209,785).  visible(br1gt_2,785). happensAt(inactive(br1gt_2),785). orientation(br1gt_3,102,785). coords(br1gt_3,301,155,785).  visible(br1gt_3,785). happensAt(walking(br1gt_3),785).
orientation(br1gt_2,12,786). coords(br1gt_2,86,209,786).  visible(br1gt_2,786). happensAt(inactive(br1gt_2),786). orientation(br1gt_3,102,786). coords(br1gt_3,301,157,786).  visible(br1gt_3,786). happensAt(walking(br1gt_3),786).
orientation(br1gt_2,12,787). coords(br1gt_2,86,209,787).  visible(br1gt_2,787). happensAt(inactive(br1gt_2),787). orientation(br1gt_3,99,787). coords(br1gt_3,301,159,787).  visible(br1gt_3,787). happensAt(walking(br1gt_3),787).
orientation(br1gt_2,12,788). coords(br1gt_2,86,209,788).  visible(br1gt_2,788). happensAt(inactive(br1gt_2),788). orientation(br1gt_3,99,788). coords(br1gt_3,300,161,788).  visible(br1gt_3,788). happensAt(walking(br1gt_3),788).
orientation(br1gt_2,12,789). coords(br1gt_2,86,209,789).  visible(br1gt_2,789). happensAt(inactive(br1gt_2),789). orientation(br1gt_3,99,789). coords(br1gt_3,300,163,789).  visible(br1gt_3,789). happensAt(walking(br1gt_3),789).
orientation(br1gt_2,12,790). coords(br1gt_2,86,209,790).  visible(br1gt_2,790). happensAt(inactive(br1gt_2),790). orientation(br1gt_3,99,790). coords(br1gt_3,299,163,790).  visible(br1gt_3,790). happensAt(walking(br1gt_3),790).
#show holdsAt/2.