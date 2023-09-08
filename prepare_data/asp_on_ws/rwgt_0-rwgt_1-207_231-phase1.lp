
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

	orientation(rwgt_0,144,207). coords(rwgt_0,189,141,207).  visible(rwgt_0,207). happensAt(walking(rwgt_0),207). orientation(rwgt_1,132,207). coords(rwgt_1,153,38,207).  visible(rwgt_1,207). happensAt(walking(rwgt_1),207).
orientation(rwgt_0,144,208). coords(rwgt_0,190,142,208).  visible(rwgt_0,208). happensAt(walking(rwgt_0),208). orientation(rwgt_1,132,208). coords(rwgt_1,153,38,208).  visible(rwgt_1,208). happensAt(walking(rwgt_1),208).
orientation(rwgt_0,144,209). coords(rwgt_0,193,142,209).  visible(rwgt_0,209). happensAt(walking(rwgt_0),209). orientation(rwgt_1,132,209). coords(rwgt_1,153,38,209).  visible(rwgt_1,209). happensAt(walking(rwgt_1),209).
orientation(rwgt_0,144,210). coords(rwgt_0,196,143,210).  visible(rwgt_0,210). happensAt(walking(rwgt_0),210). orientation(rwgt_1,132,210). coords(rwgt_1,153,38,210).  visible(rwgt_1,210). happensAt(walking(rwgt_1),210).
orientation(rwgt_0,144,211). coords(rwgt_0,200,144,211).  visible(rwgt_0,211). happensAt(walking(rwgt_0),211). orientation(rwgt_1,132,211). coords(rwgt_1,153,38,211).  visible(rwgt_1,211). happensAt(walking(rwgt_1),211).
orientation(rwgt_0,144,212). coords(rwgt_0,202,146,212).  visible(rwgt_0,212). happensAt(walking(rwgt_0),212). orientation(rwgt_1,132,212). coords(rwgt_1,153,38,212).  visible(rwgt_1,212). happensAt(walking(rwgt_1),212).
orientation(rwgt_0,144,213). coords(rwgt_0,204,147,213).  visible(rwgt_0,213). happensAt(walking(rwgt_0),213). orientation(rwgt_1,132,213). coords(rwgt_1,153,38,213).  visible(rwgt_1,213). happensAt(walking(rwgt_1),213).
orientation(rwgt_0,144,214). coords(rwgt_0,206,149,214).  visible(rwgt_0,214). happensAt(walking(rwgt_0),214). orientation(rwgt_1,132,214). coords(rwgt_1,154,38,214).  visible(rwgt_1,214). happensAt(walking(rwgt_1),214).
orientation(rwgt_0,144,215). coords(rwgt_0,208,150,215).  visible(rwgt_0,215). happensAt(walking(rwgt_0),215). orientation(rwgt_1,132,215). coords(rwgt_1,154,38,215).  visible(rwgt_1,215). happensAt(walking(rwgt_1),215).
orientation(rwgt_0,144,216). coords(rwgt_0,210,150,216).  visible(rwgt_0,216). happensAt(walking(rwgt_0),216). orientation(rwgt_1,132,216). coords(rwgt_1,154,37,216).  visible(rwgt_1,216). happensAt(walking(rwgt_1),216).
orientation(rwgt_0,144,217). coords(rwgt_0,211,151,217).  visible(rwgt_0,217). happensAt(walking(rwgt_0),217). orientation(rwgt_1,132,217). coords(rwgt_1,156,34,217).  visible(rwgt_1,217). happensAt(walking(rwgt_1),217).
orientation(rwgt_0,144,218). coords(rwgt_0,212,152,218).  visible(rwgt_0,218). happensAt(walking(rwgt_0),218). orientation(rwgt_1,132,218). coords(rwgt_1,155,34,218).  visible(rwgt_1,218). happensAt(walking(rwgt_1),218).
orientation(rwgt_0,144,219). coords(rwgt_0,213,153,219).  visible(rwgt_0,219). happensAt(walking(rwgt_0),219). orientation(rwgt_1,132,219). coords(rwgt_1,155,34,219).  visible(rwgt_1,219). happensAt(walking(rwgt_1),219).
orientation(rwgt_0,144,220). coords(rwgt_0,213,153,220).  visible(rwgt_0,220). happensAt(walking(rwgt_0),220). orientation(rwgt_1,132,220). coords(rwgt_1,156,34,220).  visible(rwgt_1,220). happensAt(walking(rwgt_1),220).
orientation(rwgt_0,144,221). coords(rwgt_0,214,155,221).  visible(rwgt_0,221). happensAt(walking(rwgt_0),221). orientation(rwgt_1,132,221). coords(rwgt_1,158,34,221).  visible(rwgt_1,221). happensAt(walking(rwgt_1),221).
orientation(rwgt_0,144,222). coords(rwgt_0,215,156,222).  visible(rwgt_0,222). happensAt(walking(rwgt_0),222). orientation(rwgt_1,132,222). coords(rwgt_1,159,34,222).  visible(rwgt_1,222). happensAt(walking(rwgt_1),222).
orientation(rwgt_0,144,223). coords(rwgt_0,216,158,223).  visible(rwgt_0,223). happensAt(walking(rwgt_0),223). orientation(rwgt_1,132,223). coords(rwgt_1,159,34,223).  visible(rwgt_1,223). happensAt(walking(rwgt_1),223).
orientation(rwgt_0,144,224). coords(rwgt_0,218,162,224).  visible(rwgt_0,224). happensAt(walking(rwgt_0),224). orientation(rwgt_1,132,224). coords(rwgt_1,159,33,224).  visible(rwgt_1,224). happensAt(walking(rwgt_1),224).
orientation(rwgt_0,144,225). coords(rwgt_0,220,165,225).  visible(rwgt_0,225). happensAt(walking(rwgt_0),225). orientation(rwgt_1,132,225). coords(rwgt_1,158,32,225).  visible(rwgt_1,225). happensAt(walking(rwgt_1),225).
orientation(rwgt_0,144,226). coords(rwgt_0,223,168,226).  visible(rwgt_0,226). happensAt(walking(rwgt_0),226). orientation(rwgt_1,132,226). coords(rwgt_1,158,31,226).  visible(rwgt_1,226). happensAt(walking(rwgt_1),226).
orientation(rwgt_0,144,227). coords(rwgt_0,225,170,227).  visible(rwgt_0,227). happensAt(walking(rwgt_0),227). orientation(rwgt_1,132,227). coords(rwgt_1,158,31,227).  visible(rwgt_1,227). happensAt(walking(rwgt_1),227).
orientation(rwgt_0,144,228). coords(rwgt_0,228,172,228).  visible(rwgt_0,228). happensAt(walking(rwgt_0),228). orientation(rwgt_1,132,228). coords(rwgt_1,159,31,228).  visible(rwgt_1,228). happensAt(walking(rwgt_1),228).
orientation(rwgt_0,144,229). coords(rwgt_0,230,174,229).  visible(rwgt_0,229). happensAt(walking(rwgt_0),229). orientation(rwgt_1,132,229). coords(rwgt_1,159,31,229).  visible(rwgt_1,229). happensAt(walking(rwgt_1),229).
orientation(rwgt_0,144,230). coords(rwgt_0,231,175,230).  visible(rwgt_0,230). happensAt(walking(rwgt_0),230). orientation(rwgt_1,132,230). coords(rwgt_1,160,31,230).  visible(rwgt_1,230). happensAt(walking(rwgt_1),230).
orientation(rwgt_0,144,231). coords(rwgt_0,233,176,231).  visible(rwgt_0,231). happensAt(walking(rwgt_0),231). orientation(rwgt_1,132,231). coords(rwgt_1,160,31,231).  visible(rwgt_1,231). happensAt(walking(rwgt_1),231).
#show holdsAt/2.