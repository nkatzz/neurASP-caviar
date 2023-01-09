
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

	orientation(rwgt_0,137,558). coords(rwgt_0,316,283,558).  visible(rwgt_0,558). happensAt(inactive(rwgt_0),558). orientation(rwgt_2,144,558). coords(rwgt_2,148,39,558).  visible(rwgt_2,558). happensAt(walking(rwgt_2),558).
orientation(rwgt_0,137,559). coords(rwgt_0,316,283,559).  visible(rwgt_0,559). happensAt(inactive(rwgt_0),559). orientation(rwgt_2,144,559). coords(rwgt_2,148,39,559).  visible(rwgt_2,559). happensAt(walking(rwgt_2),559).
orientation(rwgt_0,137,560). coords(rwgt_0,316,283,560).  visible(rwgt_0,560). happensAt(inactive(rwgt_0),560). orientation(rwgt_2,144,560). coords(rwgt_2,148,40,560).  visible(rwgt_2,560). happensAt(walking(rwgt_2),560).
orientation(rwgt_0,137,561). coords(rwgt_0,316,283,561).  visible(rwgt_0,561). happensAt(inactive(rwgt_0),561). orientation(rwgt_2,144,561). coords(rwgt_2,148,40,561).  visible(rwgt_2,561). happensAt(walking(rwgt_2),561).
orientation(rwgt_0,137,562). coords(rwgt_0,316,283,562).  visible(rwgt_0,562). happensAt(inactive(rwgt_0),562). orientation(rwgt_2,144,562). coords(rwgt_2,148,40,562).  visible(rwgt_2,562). happensAt(walking(rwgt_2),562).
orientation(rwgt_0,137,563). coords(rwgt_0,316,283,563).  visible(rwgt_0,563). happensAt(inactive(rwgt_0),563). orientation(rwgt_2,144,563). coords(rwgt_2,147,40,563).  visible(rwgt_2,563). happensAt(walking(rwgt_2),563).
orientation(rwgt_0,137,564). coords(rwgt_0,316,283,564).  visible(rwgt_0,564). happensAt(inactive(rwgt_0),564). orientation(rwgt_2,144,564). coords(rwgt_2,147,40,564).  visible(rwgt_2,564). happensAt(walking(rwgt_2),564).
orientation(rwgt_0,137,565). coords(rwgt_0,316,283,565).  visible(rwgt_0,565). happensAt(inactive(rwgt_0),565). orientation(rwgt_2,144,565). coords(rwgt_2,147,40,565).  visible(rwgt_2,565). happensAt(walking(rwgt_2),565).
orientation(rwgt_0,137,566). coords(rwgt_0,316,283,566).  visible(rwgt_0,566). happensAt(inactive(rwgt_0),566). orientation(rwgt_2,144,566). coords(rwgt_2,147,40,566).  visible(rwgt_2,566). happensAt(walking(rwgt_2),566).
orientation(rwgt_0,137,567). coords(rwgt_0,316,283,567).  visible(rwgt_0,567). happensAt(inactive(rwgt_0),567). orientation(rwgt_2,144,567). coords(rwgt_2,147,40,567).  visible(rwgt_2,567). happensAt(walking(rwgt_2),567).
orientation(rwgt_0,137,568). coords(rwgt_0,316,283,568).  visible(rwgt_0,568). happensAt(inactive(rwgt_0),568). orientation(rwgt_2,144,568). coords(rwgt_2,147,41,568).  visible(rwgt_2,568). happensAt(walking(rwgt_2),568).
orientation(rwgt_0,137,569). coords(rwgt_0,316,283,569).  visible(rwgt_0,569). happensAt(inactive(rwgt_0),569). orientation(rwgt_2,144,569). coords(rwgt_2,146,42,569).  visible(rwgt_2,569). happensAt(walking(rwgt_2),569).
orientation(rwgt_0,137,570). coords(rwgt_0,316,283,570).  visible(rwgt_0,570). happensAt(inactive(rwgt_0),570). orientation(rwgt_2,144,570). coords(rwgt_2,146,42,570).  visible(rwgt_2,570). happensAt(walking(rwgt_2),570).
orientation(rwgt_0,137,571). coords(rwgt_0,316,283,571).  visible(rwgt_0,571). happensAt(inactive(rwgt_0),571). orientation(rwgt_2,144,571). coords(rwgt_2,146,43,571).  visible(rwgt_2,571). happensAt(walking(rwgt_2),571).
orientation(rwgt_0,137,572). coords(rwgt_0,316,283,572).  visible(rwgt_0,572). happensAt(inactive(rwgt_0),572). orientation(rwgt_2,144,572). coords(rwgt_2,145,43,572).  visible(rwgt_2,572). happensAt(walking(rwgt_2),572).
orientation(rwgt_0,137,573). coords(rwgt_0,316,283,573).  visible(rwgt_0,573). happensAt(inactive(rwgt_0),573). orientation(rwgt_2,144,573). coords(rwgt_2,144,43,573).  visible(rwgt_2,573). happensAt(walking(rwgt_2),573).
orientation(rwgt_0,137,574). coords(rwgt_0,316,283,574).  visible(rwgt_0,574). happensAt(inactive(rwgt_0),574). orientation(rwgt_2,144,574). coords(rwgt_2,144,43,574).  visible(rwgt_2,574). happensAt(walking(rwgt_2),574).
orientation(rwgt_0,137,575). coords(rwgt_0,316,283,575).  visible(rwgt_0,575). happensAt(inactive(rwgt_0),575). orientation(rwgt_2,144,575). coords(rwgt_2,143,43,575).  visible(rwgt_2,575). happensAt(walking(rwgt_2),575).
orientation(rwgt_0,137,576). coords(rwgt_0,316,283,576).  visible(rwgt_0,576). happensAt(inactive(rwgt_0),576). orientation(rwgt_2,144,576). coords(rwgt_2,143,44,576).  visible(rwgt_2,576). happensAt(walking(rwgt_2),576).
orientation(rwgt_0,137,577). coords(rwgt_0,316,283,577).  visible(rwgt_0,577). happensAt(inactive(rwgt_0),577). orientation(rwgt_2,144,577). coords(rwgt_2,142,44,577).  visible(rwgt_2,577). happensAt(walking(rwgt_2),577).
orientation(rwgt_0,137,578). coords(rwgt_0,316,283,578).  visible(rwgt_0,578). happensAt(inactive(rwgt_0),578). orientation(rwgt_2,144,578). coords(rwgt_2,142,44,578).  visible(rwgt_2,578). happensAt(walking(rwgt_2),578).
orientation(rwgt_0,137,579). coords(rwgt_0,316,283,579).  visible(rwgt_0,579). happensAt(inactive(rwgt_0),579). orientation(rwgt_2,144,579). coords(rwgt_2,142,44,579).  visible(rwgt_2,579). happensAt(walking(rwgt_2),579).
orientation(rwgt_0,137,580). coords(rwgt_0,316,283,580).  visible(rwgt_0,580). happensAt(inactive(rwgt_0),580). orientation(rwgt_2,144,580). coords(rwgt_2,142,44,580).  visible(rwgt_2,580). happensAt(walking(rwgt_2),580).
orientation(rwgt_0,137,581). coords(rwgt_0,316,283,581).  visible(rwgt_0,581). happensAt(inactive(rwgt_0),581). orientation(rwgt_2,144,581). coords(rwgt_2,143,45,581).  visible(rwgt_2,581). happensAt(walking(rwgt_2),581).
orientation(rwgt_0,137,582). coords(rwgt_0,316,283,582).  visible(rwgt_0,582). happensAt(inactive(rwgt_0),582). orientation(rwgt_2,144,582). coords(rwgt_2,143,44,582).  visible(rwgt_2,582). happensAt(walking(rwgt_2),582).
#show holdsAt/2.