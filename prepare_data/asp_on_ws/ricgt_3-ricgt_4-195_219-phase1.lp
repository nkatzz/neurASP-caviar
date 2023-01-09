
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

	orientation(ricgt_3,37,195). coords(ricgt_3,285,249,195).  visible(ricgt_3,195). happensAt(walking(ricgt_3),195). orientation(ricgt_4,172,195). coords(ricgt_4,23,147,195).  visible(ricgt_4,195). happensAt(walking(ricgt_4),195).
orientation(ricgt_3,41,196). coords(ricgt_3,287,252,196).  visible(ricgt_3,196). happensAt(walking(ricgt_3),196). orientation(ricgt_4,172,196). coords(ricgt_4,23,147,196).  visible(ricgt_4,196). happensAt(walking(ricgt_4),196).
orientation(ricgt_3,45,197). coords(ricgt_3,288,253,197).  visible(ricgt_3,197). happensAt(walking(ricgt_3),197). orientation(ricgt_4,172,197). coords(ricgt_4,23,148,197).  visible(ricgt_4,197). happensAt(walking(ricgt_4),197).
orientation(ricgt_3,49,198). coords(ricgt_3,289,255,198).  visible(ricgt_3,198). happensAt(walking(ricgt_3),198). orientation(ricgt_4,172,198). coords(ricgt_4,23,148,198).  visible(ricgt_4,198). happensAt(walking(ricgt_4),198).
orientation(ricgt_3,52,199). coords(ricgt_3,290,258,199).  visible(ricgt_3,199). happensAt(walking(ricgt_3),199). orientation(ricgt_4,172,199). coords(ricgt_4,23,149,199).  visible(ricgt_4,199). happensAt(walking(ricgt_4),199).
orientation(ricgt_3,56,200). coords(ricgt_3,291,261,200).  visible(ricgt_3,200). happensAt(walking(ricgt_3),200). orientation(ricgt_4,172,200). coords(ricgt_4,23,149,200).  visible(ricgt_4,200). happensAt(walking(ricgt_4),200).
orientation(ricgt_3,60,201). coords(ricgt_3,292,264,201).  visible(ricgt_3,201). happensAt(walking(ricgt_3),201). orientation(ricgt_4,172,201). coords(ricgt_4,23,150,201).  visible(ricgt_4,201). happensAt(walking(ricgt_4),201).
orientation(ricgt_3,63,202). coords(ricgt_3,293,268,202).  visible(ricgt_3,202). happensAt(walking(ricgt_3),202). orientation(ricgt_4,172,202). coords(ricgt_4,23,150,202).  visible(ricgt_4,202). happensAt(walking(ricgt_4),202).
orientation(ricgt_3,64,203). coords(ricgt_3,294,269,203).  visible(ricgt_3,203). happensAt(walking(ricgt_3),203). orientation(ricgt_4,172,203). coords(ricgt_4,23,151,203).  visible(ricgt_4,203). happensAt(walking(ricgt_4),203).
orientation(ricgt_3,66,204). coords(ricgt_3,296,269,204).  visible(ricgt_3,204). happensAt(walking(ricgt_3),204). orientation(ricgt_4,172,204). coords(ricgt_4,23,152,204).  visible(ricgt_4,204). happensAt(walking(ricgt_4),204).
orientation(ricgt_3,68,205). coords(ricgt_3,297,269,205).  visible(ricgt_3,205). happensAt(walking(ricgt_3),205). orientation(ricgt_4,172,205). coords(ricgt_4,24,152,205).  visible(ricgt_4,205). happensAt(walking(ricgt_4),205).
orientation(ricgt_3,70,206). coords(ricgt_3,298,269,206).  visible(ricgt_3,206). happensAt(walking(ricgt_3),206). orientation(ricgt_4,172,206). coords(ricgt_4,24,152,206).  visible(ricgt_4,206). happensAt(walking(ricgt_4),206).
orientation(ricgt_3,71,207). coords(ricgt_3,299,269,207).  visible(ricgt_3,207). happensAt(walking(ricgt_3),207). orientation(ricgt_4,172,207). coords(ricgt_4,23,153,207).  visible(ricgt_4,207). happensAt(walking(ricgt_4),207).
orientation(ricgt_3,73,208). coords(ricgt_3,300,269,208).  visible(ricgt_3,208). happensAt(walking(ricgt_3),208). orientation(ricgt_4,172,208). coords(ricgt_4,23,153,208).  visible(ricgt_4,208). happensAt(walking(ricgt_4),208).
orientation(ricgt_3,76,209). coords(ricgt_3,300,270,209).  visible(ricgt_3,209). happensAt(walking(ricgt_3),209). orientation(ricgt_4,172,209). coords(ricgt_4,23,154,209).  visible(ricgt_4,209). happensAt(walking(ricgt_4),209).
orientation(ricgt_3,80,210). coords(ricgt_3,299,271,210).  visible(ricgt_3,210). happensAt(walking(ricgt_3),210). orientation(ricgt_4,172,210). coords(ricgt_4,23,154,210).  visible(ricgt_4,210). happensAt(walking(ricgt_4),210).
orientation(ricgt_3,81,211). coords(ricgt_3,300,272,211).  visible(ricgt_3,211). happensAt(walking(ricgt_3),211). orientation(ricgt_4,172,211). coords(ricgt_4,23,155,211).  visible(ricgt_4,211). happensAt(walking(ricgt_4),211).
orientation(ricgt_3,83,212). coords(ricgt_3,301,274,212).  visible(ricgt_3,212). happensAt(walking(ricgt_3),212). orientation(ricgt_4,172,212). coords(ricgt_4,23,155,212).  visible(ricgt_4,212). happensAt(walking(ricgt_4),212).
orientation(ricgt_3,85,213). coords(ricgt_3,303,276,213).  visible(ricgt_3,213). happensAt(walking(ricgt_3),213). orientation(ricgt_4,172,213). coords(ricgt_4,23,156,213).  visible(ricgt_4,213). happensAt(walking(ricgt_4),213).
orientation(ricgt_3,87,214). coords(ricgt_3,303,278,214).  visible(ricgt_3,214). happensAt(walking(ricgt_3),214). orientation(ricgt_4,172,214). coords(ricgt_4,23,156,214).  visible(ricgt_4,214). happensAt(walking(ricgt_4),214).
orientation(ricgt_3,87,215). coords(ricgt_3,305,280,215).  visible(ricgt_3,215). happensAt(walking(ricgt_3),215). orientation(ricgt_4,172,215). coords(ricgt_4,23,156,215).  visible(ricgt_4,215). happensAt(walking(ricgt_4),215).
orientation(ricgt_3,87,216). coords(ricgt_3,309,281,216).  visible(ricgt_3,216). happensAt(walking(ricgt_3),216). orientation(ricgt_4,172,216). coords(ricgt_4,23,157,216).  visible(ricgt_4,216). happensAt(walking(ricgt_4),216).
orientation(ricgt_3,87,217). coords(ricgt_3,311,281,217).  visible(ricgt_3,217). happensAt(walking(ricgt_3),217). orientation(ricgt_4,172,217). coords(ricgt_4,23,157,217).  visible(ricgt_4,217). happensAt(walking(ricgt_4),217).
orientation(ricgt_3,76,218). coords(ricgt_3,307,281,218).  visible(ricgt_3,218). happensAt(walking(ricgt_3),218). orientation(ricgt_4,172,218). coords(ricgt_4,23,158,218).  visible(ricgt_4,218). happensAt(walking(ricgt_4),218).
orientation(ricgt_3,76,219). coords(ricgt_3,308,282,219).  visible(ricgt_3,219). happensAt(walking(ricgt_3),219). orientation(ricgt_4,172,219). coords(ricgt_4,23,158,219).  visible(ricgt_4,219). happensAt(walking(ricgt_4),219).
#show holdsAt/2.