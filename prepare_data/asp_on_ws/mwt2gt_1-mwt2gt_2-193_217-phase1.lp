
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

	orientation(mwt2gt_1,141,193). coords(mwt2gt_1,156,41,193).  visible(mwt2gt_1,193). happensAt(walking(mwt2gt_1),193). orientation(mwt2gt_2,25,193). coords(mwt2gt_2,247,257,193).  visible(mwt2gt_2,193). happensAt(walking(mwt2gt_2),193).
orientation(mwt2gt_1,141,194). coords(mwt2gt_1,155,41,194).  visible(mwt2gt_1,194). happensAt(walking(mwt2gt_1),194). orientation(mwt2gt_2,25,194). coords(mwt2gt_2,246,252,194).  visible(mwt2gt_2,194). happensAt(walking(mwt2gt_2),194).
orientation(mwt2gt_1,141,195). coords(mwt2gt_1,154,41,195).  visible(mwt2gt_1,195). happensAt(walking(mwt2gt_1),195). orientation(mwt2gt_2,25,195). coords(mwt2gt_2,245,248,195).  visible(mwt2gt_2,195). happensAt(walking(mwt2gt_2),195).
orientation(mwt2gt_1,141,196). coords(mwt2gt_1,154,41,196).  visible(mwt2gt_1,196). happensAt(walking(mwt2gt_1),196). orientation(mwt2gt_2,25,196). coords(mwt2gt_2,245,245,196).  visible(mwt2gt_2,196). happensAt(walking(mwt2gt_2),196).
orientation(mwt2gt_1,141,197). coords(mwt2gt_1,154,41,197).  visible(mwt2gt_1,197). happensAt(walking(mwt2gt_1),197). orientation(mwt2gt_2,19,197). coords(mwt2gt_2,245,245,197).  visible(mwt2gt_2,197). happensAt(walking(mwt2gt_2),197).
orientation(mwt2gt_1,141,198). coords(mwt2gt_1,153,42,198).  visible(mwt2gt_1,198). happensAt(walking(mwt2gt_1),198). orientation(mwt2gt_2,15,198). coords(mwt2gt_2,244,244,198).  visible(mwt2gt_2,198). happensAt(walking(mwt2gt_2),198).
orientation(mwt2gt_1,141,199). coords(mwt2gt_1,153,43,199).  visible(mwt2gt_1,199). happensAt(walking(mwt2gt_1),199). orientation(mwt2gt_2,15,199). coords(mwt2gt_2,244,244,199).  visible(mwt2gt_2,199). happensAt(walking(mwt2gt_2),199).
orientation(mwt2gt_1,141,200). coords(mwt2gt_1,153,45,200).  visible(mwt2gt_1,200). happensAt(walking(mwt2gt_1),200). orientation(mwt2gt_2,15,200). coords(mwt2gt_2,243,243,200).  visible(mwt2gt_2,200). happensAt(walking(mwt2gt_2),200).
orientation(mwt2gt_1,141,201). coords(mwt2gt_1,153,46,201).  visible(mwt2gt_1,201). happensAt(walking(mwt2gt_1),201). orientation(mwt2gt_2,0,201). coords(mwt2gt_2,243,241,201).  visible(mwt2gt_2,201). happensAt(walking(mwt2gt_2),201).
orientation(mwt2gt_1,141,202). coords(mwt2gt_1,153,47,202).  visible(mwt2gt_1,202). happensAt(walking(mwt2gt_1),202). orientation(mwt2gt_2,0,202). coords(mwt2gt_2,242,239,202).  visible(mwt2gt_2,202). happensAt(walking(mwt2gt_2),202).
orientation(mwt2gt_1,141,203). coords(mwt2gt_1,152,48,203).  visible(mwt2gt_1,203). happensAt(walking(mwt2gt_1),203). orientation(mwt2gt_2,174,203). coords(mwt2gt_2,242,237,203).  visible(mwt2gt_2,203). happensAt(walking(mwt2gt_2),203).
orientation(mwt2gt_1,141,204). coords(mwt2gt_1,152,48,204).  visible(mwt2gt_1,204). happensAt(walking(mwt2gt_1),204). orientation(mwt2gt_2,174,204). coords(mwt2gt_2,242,234,204).  visible(mwt2gt_2,204). happensAt(walking(mwt2gt_2),204).
orientation(mwt2gt_1,141,205). coords(mwt2gt_1,152,49,205).  visible(mwt2gt_1,205). happensAt(walking(mwt2gt_1),205). orientation(mwt2gt_2,177,205). coords(mwt2gt_2,241,231,205).  visible(mwt2gt_2,205). happensAt(walking(mwt2gt_2),205).
orientation(mwt2gt_1,141,206). coords(mwt2gt_1,151,49,206).  visible(mwt2gt_1,206). happensAt(walking(mwt2gt_1),206). orientation(mwt2gt_2,177,206). coords(mwt2gt_2,241,229,206).  visible(mwt2gt_2,206). happensAt(walking(mwt2gt_2),206).
orientation(mwt2gt_1,141,207). coords(mwt2gt_1,151,49,207).  visible(mwt2gt_1,207). happensAt(walking(mwt2gt_1),207). orientation(mwt2gt_2,177,207). coords(mwt2gt_2,241,227,207).  visible(mwt2gt_2,207). happensAt(walking(mwt2gt_2),207).
orientation(mwt2gt_1,141,208). coords(mwt2gt_1,150,49,208).  visible(mwt2gt_1,208). happensAt(walking(mwt2gt_1),208). orientation(mwt2gt_2,177,208). coords(mwt2gt_2,240,224,208).  visible(mwt2gt_2,208). happensAt(walking(mwt2gt_2),208).
orientation(mwt2gt_1,141,209). coords(mwt2gt_1,149,49,209).  visible(mwt2gt_1,209). happensAt(walking(mwt2gt_1),209). orientation(mwt2gt_2,177,209). coords(mwt2gt_2,240,222,209).  visible(mwt2gt_2,209). happensAt(walking(mwt2gt_2),209).
orientation(mwt2gt_1,141,210). coords(mwt2gt_1,149,49,210).  visible(mwt2gt_1,210). happensAt(walking(mwt2gt_1),210). orientation(mwt2gt_2,177,210). coords(mwt2gt_2,239,219,210).  visible(mwt2gt_2,210). happensAt(walking(mwt2gt_2),210).
orientation(mwt2gt_1,141,211). coords(mwt2gt_1,147,49,211).  visible(mwt2gt_1,211). happensAt(walking(mwt2gt_1),211). orientation(mwt2gt_2,177,211). coords(mwt2gt_2,239,219,211).  visible(mwt2gt_2,211). happensAt(walking(mwt2gt_2),211).
orientation(mwt2gt_1,141,212). coords(mwt2gt_1,147,49,212).  visible(mwt2gt_1,212). happensAt(walking(mwt2gt_1),212). orientation(mwt2gt_2,174,212). coords(mwt2gt_2,238,219,212).  visible(mwt2gt_2,212). happensAt(walking(mwt2gt_2),212).
orientation(mwt2gt_1,141,213). coords(mwt2gt_1,147,49,213).  visible(mwt2gt_1,213). happensAt(walking(mwt2gt_1),213). orientation(mwt2gt_2,174,213). coords(mwt2gt_2,237,218,213).  visible(mwt2gt_2,213). happensAt(walking(mwt2gt_2),213).
orientation(mwt2gt_1,141,214). coords(mwt2gt_1,147,49,214).  visible(mwt2gt_1,214). happensAt(walking(mwt2gt_1),214). orientation(mwt2gt_2,174,214). coords(mwt2gt_2,235,216,214).  visible(mwt2gt_2,214). happensAt(walking(mwt2gt_2),214).
orientation(mwt2gt_1,141,215). coords(mwt2gt_1,147,49,215).  visible(mwt2gt_1,215). happensAt(walking(mwt2gt_1),215). orientation(mwt2gt_2,174,215). coords(mwt2gt_2,234,214,215).  visible(mwt2gt_2,215). happensAt(walking(mwt2gt_2),215).
orientation(mwt2gt_1,141,216). coords(mwt2gt_1,147,51,216).  visible(mwt2gt_1,216). happensAt(walking(mwt2gt_1),216). orientation(mwt2gt_2,168,216). coords(mwt2gt_2,233,211,216).  visible(mwt2gt_2,216). happensAt(walking(mwt2gt_2),216).
orientation(mwt2gt_1,141,217). coords(mwt2gt_1,147,52,217).  visible(mwt2gt_1,217). happensAt(walking(mwt2gt_1),217). orientation(mwt2gt_2,168,217). coords(mwt2gt_2,231,207,217).  visible(mwt2gt_2,217). happensAt(walking(mwt2gt_2),217).
#show holdsAt/2.