
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

	orientation(lbpugt_2,97,755). coords(lbpugt_2,295,98,755).  visible(lbpugt_2,755). happensAt(active(lbpugt_2),755). orientation(lbpugt_4,135,755). coords(lbpugt_4,219,34,755).  visible(lbpugt_4,755). happensAt(inactive(lbpugt_4),755).
orientation(lbpugt_2,97,756). coords(lbpugt_2,295,99,756).  visible(lbpugt_2,756). happensAt(active(lbpugt_2),756). orientation(lbpugt_4,135,756). coords(lbpugt_4,219,34,756).  visible(lbpugt_4,756). happensAt(inactive(lbpugt_4),756).
orientation(lbpugt_2,97,757). coords(lbpugt_2,295,99,757).  visible(lbpugt_2,757). happensAt(active(lbpugt_2),757). orientation(lbpugt_4,135,757). coords(lbpugt_4,219,34,757).  visible(lbpugt_4,757). happensAt(inactive(lbpugt_4),757).
orientation(lbpugt_2,97,758). coords(lbpugt_2,295,99,758).  visible(lbpugt_2,758). happensAt(active(lbpugt_2),758). orientation(lbpugt_4,135,758). coords(lbpugt_4,219,34,758).  visible(lbpugt_4,758). happensAt(inactive(lbpugt_4),758).
orientation(lbpugt_2,97,759). coords(lbpugt_2,295,99,759).  visible(lbpugt_2,759). happensAt(active(lbpugt_2),759). orientation(lbpugt_4,135,759). coords(lbpugt_4,219,34,759).  visible(lbpugt_4,759). happensAt(inactive(lbpugt_4),759).
orientation(lbpugt_2,97,760). coords(lbpugt_2,295,99,760).  visible(lbpugt_2,760). happensAt(active(lbpugt_2),760). orientation(lbpugt_4,135,760). coords(lbpugt_4,219,34,760).  visible(lbpugt_4,760). happensAt(inactive(lbpugt_4),760).
orientation(lbpugt_2,97,761). coords(lbpugt_2,295,99,761).  visible(lbpugt_2,761). happensAt(active(lbpugt_2),761). orientation(lbpugt_4,135,761). coords(lbpugt_4,219,34,761).  visible(lbpugt_4,761). happensAt(inactive(lbpugt_4),761).
orientation(lbpugt_2,97,762). coords(lbpugt_2,295,99,762).  visible(lbpugt_2,762). happensAt(active(lbpugt_2),762). orientation(lbpugt_4,135,762). coords(lbpugt_4,219,34,762).  visible(lbpugt_4,762). happensAt(inactive(lbpugt_4),762).
orientation(lbpugt_2,97,763). coords(lbpugt_2,295,99,763).  visible(lbpugt_2,763). happensAt(active(lbpugt_2),763). orientation(lbpugt_4,135,763). coords(lbpugt_4,219,34,763).  visible(lbpugt_4,763). happensAt(inactive(lbpugt_4),763).
orientation(lbpugt_2,97,764). coords(lbpugt_2,295,99,764).  visible(lbpugt_2,764). happensAt(active(lbpugt_2),764). orientation(lbpugt_4,135,764). coords(lbpugt_4,219,34,764).  visible(lbpugt_4,764). happensAt(inactive(lbpugt_4),764).
orientation(lbpugt_2,97,765). coords(lbpugt_2,295,99,765).  visible(lbpugt_2,765). happensAt(active(lbpugt_2),765). orientation(lbpugt_4,135,765). coords(lbpugt_4,219,34,765).  visible(lbpugt_4,765). happensAt(inactive(lbpugt_4),765).
orientation(lbpugt_2,97,766). coords(lbpugt_2,295,99,766).  visible(lbpugt_2,766). happensAt(active(lbpugt_2),766). orientation(lbpugt_4,135,766). coords(lbpugt_4,219,34,766).  visible(lbpugt_4,766). happensAt(inactive(lbpugt_4),766).
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
#show holdsAt/2.