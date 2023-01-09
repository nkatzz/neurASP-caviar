
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

	orientation(mws1gt_3,154,190). coords(mws1gt_3,152,154,190).  visible(mws1gt_3,190). happensAt(walking(mws1gt_3),190). orientation(mws1gt_5,145,190). coords(mws1gt_5,55,99,190).  visible(mws1gt_5,190). happensAt(walking(mws1gt_5),190).
orientation(mws1gt_3,158,191). coords(mws1gt_3,152,154,191).  visible(mws1gt_3,191). happensAt(walking(mws1gt_3),191). orientation(mws1gt_5,145,191). coords(mws1gt_5,54,99,191).  visible(mws1gt_5,191). happensAt(walking(mws1gt_5),191).
orientation(mws1gt_3,160,192). coords(mws1gt_3,151,154,192).  visible(mws1gt_3,192). happensAt(walking(mws1gt_3),192). orientation(mws1gt_5,145,192). coords(mws1gt_5,53,99,192).  visible(mws1gt_5,192). happensAt(walking(mws1gt_5),192).
orientation(mws1gt_3,163,193). coords(mws1gt_3,150,154,193).  visible(mws1gt_3,193). happensAt(walking(mws1gt_3),193). orientation(mws1gt_5,145,193). coords(mws1gt_5,53,99,193).  visible(mws1gt_5,193). happensAt(walking(mws1gt_5),193).
orientation(mws1gt_3,162,194). coords(mws1gt_3,149,154,194).  visible(mws1gt_3,194). happensAt(walking(mws1gt_3),194). orientation(mws1gt_5,145,194). coords(mws1gt_5,49,99,194).  visible(mws1gt_5,194). happensAt(walking(mws1gt_5),194).
orientation(mws1gt_3,161,195). coords(mws1gt_3,147,154,195).  visible(mws1gt_3,195). happensAt(walking(mws1gt_3),195). orientation(mws1gt_5,145,195). coords(mws1gt_5,49,99,195).  visible(mws1gt_5,195). happensAt(walking(mws1gt_5),195).
orientation(mws1gt_3,162,196). coords(mws1gt_3,146,154,196).  visible(mws1gt_3,196). happensAt(walking(mws1gt_3),196). orientation(mws1gt_5,145,196). coords(mws1gt_5,49,99,196).  visible(mws1gt_5,196). happensAt(walking(mws1gt_5),196).
orientation(mws1gt_3,162,197). coords(mws1gt_3,144,154,197).  visible(mws1gt_3,197). happensAt(walking(mws1gt_3),197). orientation(mws1gt_5,158,197). coords(mws1gt_5,48,100,197).  visible(mws1gt_5,197). happensAt(walking(mws1gt_5),197).
orientation(mws1gt_3,162,198). coords(mws1gt_3,143,154,198).  visible(mws1gt_3,198). happensAt(walking(mws1gt_3),198). orientation(mws1gt_5,158,198). coords(mws1gt_5,48,100,198).  visible(mws1gt_5,198). happensAt(walking(mws1gt_5),198).
orientation(mws1gt_3,162,199). coords(mws1gt_3,141,154,199).  visible(mws1gt_3,199). happensAt(walking(mws1gt_3),199). orientation(mws1gt_5,161,199). coords(mws1gt_5,48,102,199).  visible(mws1gt_5,199). happensAt(walking(mws1gt_5),199).
orientation(mws1gt_3,162,200). coords(mws1gt_3,140,154,200).  visible(mws1gt_3,200). happensAt(walking(mws1gt_3),200). orientation(mws1gt_5,161,200). coords(mws1gt_5,48,102,200).  visible(mws1gt_5,200). happensAt(walking(mws1gt_5),200).
orientation(mws1gt_3,162,201). coords(mws1gt_3,140,154,201).  visible(mws1gt_3,201). happensAt(walking(mws1gt_3),201). orientation(mws1gt_5,161,201). coords(mws1gt_5,48,102,201).  visible(mws1gt_5,201). happensAt(walking(mws1gt_5),201).
orientation(mws1gt_3,162,202). coords(mws1gt_3,139,155,202).  visible(mws1gt_3,202). happensAt(walking(mws1gt_3),202). orientation(mws1gt_5,161,202). coords(mws1gt_5,47,103,202).  visible(mws1gt_5,202). happensAt(walking(mws1gt_5),202).
orientation(mws1gt_3,162,203). coords(mws1gt_3,139,155,203).  visible(mws1gt_3,203). happensAt(walking(mws1gt_3),203). orientation(mws1gt_5,161,203). coords(mws1gt_5,46,104,203).  visible(mws1gt_5,203). happensAt(walking(mws1gt_5),203).
orientation(mws1gt_3,162,204). coords(mws1gt_3,139,155,204).  visible(mws1gt_3,204). happensAt(walking(mws1gt_3),204). orientation(mws1gt_5,161,204). coords(mws1gt_5,45,105,204).  visible(mws1gt_5,204). happensAt(walking(mws1gt_5),204).
orientation(mws1gt_3,162,205). coords(mws1gt_3,139,155,205).  visible(mws1gt_3,205). happensAt(walking(mws1gt_3),205). orientation(mws1gt_5,161,205). coords(mws1gt_5,45,107,205).  visible(mws1gt_5,205). happensAt(walking(mws1gt_5),205).
orientation(mws1gt_3,162,206). coords(mws1gt_3,138,155,206).  visible(mws1gt_3,206). happensAt(walking(mws1gt_3),206). orientation(mws1gt_5,161,206). coords(mws1gt_5,45,107,206).  visible(mws1gt_5,206). happensAt(walking(mws1gt_5),206).
orientation(mws1gt_3,162,207). coords(mws1gt_3,138,155,207).  visible(mws1gt_3,207). happensAt(walking(mws1gt_3),207). orientation(mws1gt_5,161,207). coords(mws1gt_5,45,107,207).  visible(mws1gt_5,207). happensAt(walking(mws1gt_5),207).
orientation(mws1gt_3,162,208). coords(mws1gt_3,137,155,208).  visible(mws1gt_3,208). happensAt(walking(mws1gt_3),208). orientation(mws1gt_5,161,208). coords(mws1gt_5,45,108,208).  visible(mws1gt_5,208). happensAt(walking(mws1gt_5),208).
orientation(mws1gt_3,162,209). coords(mws1gt_3,135,155,209).  visible(mws1gt_3,209). happensAt(walking(mws1gt_3),209). orientation(mws1gt_5,161,209). coords(mws1gt_5,45,109,209).  visible(mws1gt_5,209). happensAt(walking(mws1gt_5),209).
orientation(mws1gt_3,162,210). coords(mws1gt_3,134,155,210).  visible(mws1gt_3,210). happensAt(walking(mws1gt_3),210). orientation(mws1gt_5,161,210). coords(mws1gt_5,45,109,210).  visible(mws1gt_5,210). happensAt(walking(mws1gt_5),210).
orientation(mws1gt_3,162,211). coords(mws1gt_3,132,155,211).  visible(mws1gt_3,211). happensAt(walking(mws1gt_3),211). orientation(mws1gt_5,161,211). coords(mws1gt_5,45,109,211).  visible(mws1gt_5,211). happensAt(walking(mws1gt_5),211).
orientation(mws1gt_3,162,212). coords(mws1gt_3,130,155,212).  visible(mws1gt_3,212). happensAt(walking(mws1gt_3),212). orientation(mws1gt_5,161,212). coords(mws1gt_5,45,110,212).  visible(mws1gt_5,212). happensAt(walking(mws1gt_5),212).
orientation(mws1gt_3,162,213). coords(mws1gt_3,127,155,213).  visible(mws1gt_3,213). happensAt(walking(mws1gt_3),213). orientation(mws1gt_5,161,213). coords(mws1gt_5,45,111,213).  visible(mws1gt_5,213). happensAt(walking(mws1gt_5),213).
orientation(mws1gt_3,162,214). coords(mws1gt_3,126,154,214).  visible(mws1gt_3,214). happensAt(walking(mws1gt_3),214). orientation(mws1gt_5,161,214). coords(mws1gt_5,45,113,214).  visible(mws1gt_5,214). happensAt(walking(mws1gt_5),214).
#show holdsAt/2.