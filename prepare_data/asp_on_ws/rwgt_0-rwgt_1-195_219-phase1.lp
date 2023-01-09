
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

	orientation(rwgt_0,149,195). coords(rwgt_0,177,127,195).  visible(rwgt_0,195). happensAt(active(rwgt_0),195). orientation(rwgt_1,135,195). coords(rwgt_1,141,42,195).  visible(rwgt_1,195). happensAt(walking(rwgt_1),195).
orientation(rwgt_0,149,196). coords(rwgt_0,179,129,196).  visible(rwgt_0,196). happensAt(active(rwgt_0),196). orientation(rwgt_1,135,196). coords(rwgt_1,142,42,196).  visible(rwgt_1,196). happensAt(walking(rwgt_1),196).
orientation(rwgt_0,149,197). coords(rwgt_0,180,129,197).  visible(rwgt_0,197). happensAt(active(rwgt_0),197). orientation(rwgt_1,135,197). coords(rwgt_1,142,40,197).  visible(rwgt_1,197). happensAt(walking(rwgt_1),197).
orientation(rwgt_0,144,198). coords(rwgt_0,181,133,198).  visible(rwgt_0,198). happensAt(active(rwgt_0),198). orientation(rwgt_1,135,198). coords(rwgt_1,143,39,198).  visible(rwgt_1,198). happensAt(walking(rwgt_1),198).
orientation(rwgt_0,144,199). coords(rwgt_0,183,135,199).  visible(rwgt_0,199). happensAt(active(rwgt_0),199). orientation(rwgt_1,135,199). coords(rwgt_1,143,39,199).  visible(rwgt_1,199). happensAt(walking(rwgt_1),199).
orientation(rwgt_0,144,200). coords(rwgt_0,184,136,200).  visible(rwgt_0,200). happensAt(active(rwgt_0),200). orientation(rwgt_1,135,200). coords(rwgt_1,145,39,200).  visible(rwgt_1,200). happensAt(walking(rwgt_1),200).
orientation(rwgt_0,144,201). coords(rwgt_0,185,137,201).  visible(rwgt_0,201). happensAt(active(rwgt_0),201). orientation(rwgt_1,135,201). coords(rwgt_1,145,39,201).  visible(rwgt_1,201). happensAt(walking(rwgt_1),201).
orientation(rwgt_0,144,202). coords(rwgt_0,186,138,202).  visible(rwgt_0,202). happensAt(active(rwgt_0),202). orientation(rwgt_1,135,202). coords(rwgt_1,146,38,202).  visible(rwgt_1,202). happensAt(walking(rwgt_1),202).
orientation(rwgt_0,144,203). coords(rwgt_0,187,139,203).  visible(rwgt_0,203). happensAt(walking(rwgt_0),203). orientation(rwgt_1,134,203). coords(rwgt_1,147,38,203).  visible(rwgt_1,203). happensAt(walking(rwgt_1),203).
orientation(rwgt_0,144,204). coords(rwgt_0,187,140,204).  visible(rwgt_0,204). happensAt(walking(rwgt_0),204). orientation(rwgt_1,133,204). coords(rwgt_1,149,38,204).  visible(rwgt_1,204). happensAt(walking(rwgt_1),204).
orientation(rwgt_0,144,205). coords(rwgt_0,189,140,205).  visible(rwgt_0,205). happensAt(walking(rwgt_0),205). orientation(rwgt_1,132,205). coords(rwgt_1,152,38,205).  visible(rwgt_1,205). happensAt(walking(rwgt_1),205).
orientation(rwgt_0,144,206). coords(rwgt_0,189,141,206).  visible(rwgt_0,206). happensAt(walking(rwgt_0),206). orientation(rwgt_1,132,206). coords(rwgt_1,152,38,206).  visible(rwgt_1,206). happensAt(walking(rwgt_1),206).
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
#show holdsAt/2.