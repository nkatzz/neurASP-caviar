
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

	orientation(mws1gt_1,155,192). coords(mws1gt_1,144,136,192).  visible(mws1gt_1,192). happensAt(walking(mws1gt_1),192). orientation(mws1gt_6,140,192). coords(mws1gt_6,66,81,192).  visible(mws1gt_6,192). happensAt(walking(mws1gt_6),192).
orientation(mws1gt_1,155,193). coords(mws1gt_1,144,136,193).  visible(mws1gt_1,193). happensAt(walking(mws1gt_1),193). orientation(mws1gt_6,140,193). coords(mws1gt_6,66,82,193).  visible(mws1gt_6,193). happensAt(walking(mws1gt_6),193).
orientation(mws1gt_1,155,194). coords(mws1gt_1,143,136,194).  visible(mws1gt_1,194). happensAt(walking(mws1gt_1),194). orientation(mws1gt_6,140,194). coords(mws1gt_6,66,83,194).  visible(mws1gt_6,194). happensAt(walking(mws1gt_6),194).
orientation(mws1gt_1,155,195). coords(mws1gt_1,143,135,195).  visible(mws1gt_1,195). happensAt(walking(mws1gt_1),195). orientation(mws1gt_6,140,195). coords(mws1gt_6,66,84,195).  visible(mws1gt_6,195). happensAt(walking(mws1gt_6),195).
orientation(mws1gt_1,155,196). coords(mws1gt_1,143,135,196).  visible(mws1gt_1,196). happensAt(walking(mws1gt_1),196). orientation(mws1gt_6,140,196). coords(mws1gt_6,65,84,196).  visible(mws1gt_6,196). happensAt(walking(mws1gt_6),196).
orientation(mws1gt_1,155,197). coords(mws1gt_1,142,135,197).  visible(mws1gt_1,197). happensAt(walking(mws1gt_1),197). orientation(mws1gt_6,140,197). coords(mws1gt_6,64,84,197).  visible(mws1gt_6,197). happensAt(walking(mws1gt_6),197).
orientation(mws1gt_1,155,198). coords(mws1gt_1,142,134,198).  visible(mws1gt_1,198). happensAt(walking(mws1gt_1),198). orientation(mws1gt_6,140,198). coords(mws1gt_6,64,84,198).  visible(mws1gt_6,198). happensAt(walking(mws1gt_6),198).
orientation(mws1gt_1,155,199). coords(mws1gt_1,142,134,199).  visible(mws1gt_1,199). happensAt(walking(mws1gt_1),199). orientation(mws1gt_6,140,199). coords(mws1gt_6,64,85,199).  visible(mws1gt_6,199). happensAt(walking(mws1gt_6),199).
orientation(mws1gt_1,155,200). coords(mws1gt_1,140,133,200).  visible(mws1gt_1,200). happensAt(walking(mws1gt_1),200). orientation(mws1gt_6,140,200). coords(mws1gt_6,63,85,200).  visible(mws1gt_6,200). happensAt(walking(mws1gt_6),200).
orientation(mws1gt_1,155,201). coords(mws1gt_1,139,133,201).  visible(mws1gt_1,201). happensAt(walking(mws1gt_1),201). orientation(mws1gt_6,140,201). coords(mws1gt_6,62,86,201).  visible(mws1gt_6,201). happensAt(walking(mws1gt_6),201).
orientation(mws1gt_1,155,202). coords(mws1gt_1,139,133,202).  visible(mws1gt_1,202). happensAt(walking(mws1gt_1),202). orientation(mws1gt_6,140,202). coords(mws1gt_6,61,86,202).  visible(mws1gt_6,202). happensAt(walking(mws1gt_6),202).
orientation(mws1gt_1,155,203). coords(mws1gt_1,138,133,203).  visible(mws1gt_1,203). happensAt(walking(mws1gt_1),203). orientation(mws1gt_6,140,203). coords(mws1gt_6,60,87,203).  visible(mws1gt_6,203). happensAt(walking(mws1gt_6),203).
orientation(mws1gt_1,155,204). coords(mws1gt_1,138,133,204).  visible(mws1gt_1,204). happensAt(walking(mws1gt_1),204). orientation(mws1gt_6,140,204). coords(mws1gt_6,60,88,204).  visible(mws1gt_6,204). happensAt(walking(mws1gt_6),204).
orientation(mws1gt_1,155,205). coords(mws1gt_1,138,133,205).  visible(mws1gt_1,205). happensAt(walking(mws1gt_1),205). orientation(mws1gt_6,146,205). coords(mws1gt_6,60,90,205).  visible(mws1gt_6,205). happensAt(walking(mws1gt_6),205).
orientation(mws1gt_1,155,206). coords(mws1gt_1,137,132,206).  visible(mws1gt_1,206). happensAt(walking(mws1gt_1),206). orientation(mws1gt_6,146,206). coords(mws1gt_6,59,91,206).  visible(mws1gt_6,206). happensAt(walking(mws1gt_6),206).
orientation(mws1gt_1,155,207). coords(mws1gt_1,137,132,207).  visible(mws1gt_1,207). happensAt(walking(mws1gt_1),207). orientation(mws1gt_6,146,207). coords(mws1gt_6,59,92,207).  visible(mws1gt_6,207). happensAt(walking(mws1gt_6),207).
orientation(mws1gt_1,155,208). coords(mws1gt_1,137,132,208).  visible(mws1gt_1,208). happensAt(walking(mws1gt_1),208). orientation(mws1gt_6,146,208). coords(mws1gt_6,59,92,208).  visible(mws1gt_6,208). happensAt(walking(mws1gt_6),208).
orientation(mws1gt_1,155,209). coords(mws1gt_1,136,132,209).  visible(mws1gt_1,209). happensAt(walking(mws1gt_1),209). orientation(mws1gt_6,146,209). coords(mws1gt_6,59,94,209).  visible(mws1gt_6,209). happensAt(walking(mws1gt_6),209).
orientation(mws1gt_1,155,210). coords(mws1gt_1,135,132,210).  visible(mws1gt_1,210). happensAt(walking(mws1gt_1),210). orientation(mws1gt_6,146,210). coords(mws1gt_6,59,94,210).  visible(mws1gt_6,210). happensAt(walking(mws1gt_6),210).
orientation(mws1gt_1,155,211). coords(mws1gt_1,134,132,211).  visible(mws1gt_1,211). happensAt(walking(mws1gt_1),211). orientation(mws1gt_6,146,211). coords(mws1gt_6,59,94,211).  visible(mws1gt_6,211). happensAt(walking(mws1gt_6),211).
orientation(mws1gt_1,155,212). coords(mws1gt_1,134,132,212).  visible(mws1gt_1,212). happensAt(walking(mws1gt_1),212). orientation(mws1gt_6,146,212). coords(mws1gt_6,59,95,212).  visible(mws1gt_6,212). happensAt(walking(mws1gt_6),212).
orientation(mws1gt_1,155,213). coords(mws1gt_1,132,132,213).  visible(mws1gt_1,213). happensAt(walking(mws1gt_1),213). orientation(mws1gt_6,146,213). coords(mws1gt_6,58,95,213).  visible(mws1gt_6,213). happensAt(walking(mws1gt_6),213).
orientation(mws1gt_1,155,214). coords(mws1gt_1,131,131,214).  visible(mws1gt_1,214). happensAt(walking(mws1gt_1),214). orientation(mws1gt_6,146,214). coords(mws1gt_6,57,95,214).  visible(mws1gt_6,214). happensAt(walking(mws1gt_6),214).
orientation(mws1gt_1,155,215). coords(mws1gt_1,130,131,215).  visible(mws1gt_1,215). happensAt(walking(mws1gt_1),215). orientation(mws1gt_6,146,215). coords(mws1gt_6,56,96,215).  visible(mws1gt_6,215). happensAt(walking(mws1gt_6),215).
orientation(mws1gt_1,155,216). coords(mws1gt_1,128,132,216).  visible(mws1gt_1,216). happensAt(walking(mws1gt_1),216). orientation(mws1gt_6,146,216). coords(mws1gt_6,55,96,216).  visible(mws1gt_6,216). happensAt(walking(mws1gt_6),216).
#show holdsAt/2.