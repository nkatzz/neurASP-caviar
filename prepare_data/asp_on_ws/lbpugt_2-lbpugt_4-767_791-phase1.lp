
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

	orientation(lbpugt_2,97,767). coords(lbpugt_2,295,99,767).  visible(lbpugt_2,767). happensAt(active(lbpugt_2),767). orientation(lbpugt_4,135,767). coords(lbpugt_4,219,34,767).  visible(lbpugt_4,767). happensAt(inactive(lbpugt_4),767).
orientation(lbpugt_2,97,768). coords(lbpugt_2,295,99,768).  visible(lbpugt_2,768). happensAt(active(lbpugt_2),768). orientation(lbpugt_4,135,768). coords(lbpugt_4,219,34,768).  visible(lbpugt_4,768). happensAt(inactive(lbpugt_4),768).
orientation(lbpugt_2,97,769). coords(lbpugt_2,295,99,769).  visible(lbpugt_2,769). happensAt(active(lbpugt_2),769). orientation(lbpugt_4,135,769). coords(lbpugt_4,219,34,769).  visible(lbpugt_4,769). happensAt(inactive(lbpugt_4),769).
orientation(lbpugt_2,97,770). coords(lbpugt_2,295,99,770).  visible(lbpugt_2,770). happensAt(active(lbpugt_2),770). orientation(lbpugt_4,135,770). coords(lbpugt_4,219,34,770).  visible(lbpugt_4,770). happensAt(inactive(lbpugt_4),770).
orientation(lbpugt_2,97,771). coords(lbpugt_2,295,99,771).  visible(lbpugt_2,771). happensAt(active(lbpugt_2),771). orientation(lbpugt_4,135,771). coords(lbpugt_4,219,34,771).  visible(lbpugt_4,771). happensAt(inactive(lbpugt_4),771).
orientation(lbpugt_2,97,772). coords(lbpugt_2,295,99,772).  visible(lbpugt_2,772). happensAt(active(lbpugt_2),772). orientation(lbpugt_4,135,772). coords(lbpugt_4,219,34,772).  visible(lbpugt_4,772). happensAt(inactive(lbpugt_4),772).
orientation(lbpugt_2,97,773). coords(lbpugt_2,295,99,773).  visible(lbpugt_2,773). happensAt(active(lbpugt_2),773). orientation(lbpugt_4,135,773). coords(lbpugt_4,219,34,773).  visible(lbpugt_4,773). happensAt(inactive(lbpugt_4),773).
orientation(lbpugt_2,97,774). coords(lbpugt_2,295,99,774).  visible(lbpugt_2,774). happensAt(active(lbpugt_2),774). orientation(lbpugt_4,135,774). coords(lbpugt_4,219,34,774).  visible(lbpugt_4,774). happensAt(inactive(lbpugt_4),774).
orientation(lbpugt_2,97,775). coords(lbpugt_2,295,99,775).  visible(lbpugt_2,775). happensAt(active(lbpugt_2),775). orientation(lbpugt_4,135,775). coords(lbpugt_4,219,34,775).  visible(lbpugt_4,775). happensAt(inactive(lbpugt_4),775).
orientation(lbpugt_2,97,776). coords(lbpugt_2,295,99,776).  visible(lbpugt_2,776). happensAt(active(lbpugt_2),776). orientation(lbpugt_4,135,776). coords(lbpugt_4,219,34,776).  visible(lbpugt_4,776). happensAt(inactive(lbpugt_4),776).
orientation(lbpugt_2,97,777). coords(lbpugt_2,295,99,777).  visible(lbpugt_2,777). happensAt(active(lbpugt_2),777). orientation(lbpugt_4,135,777). coords(lbpugt_4,219,34,777).  visible(lbpugt_4,777). happensAt(inactive(lbpugt_4),777).
orientation(lbpugt_2,97,778). coords(lbpugt_2,295,99,778).  visible(lbpugt_2,778). happensAt(active(lbpugt_2),778). orientation(lbpugt_4,135,778). coords(lbpugt_4,219,34,778).  visible(lbpugt_4,778). happensAt(inactive(lbpugt_4),778).
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
#show holdsAt/2.