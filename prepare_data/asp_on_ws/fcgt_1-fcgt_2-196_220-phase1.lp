
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

	orientation(fcgt_1,32,196). coords(fcgt_1,250,273,196).  visible(fcgt_1,196). happensAt(walking(fcgt_1),196). orientation(fcgt_2,166,196). coords(fcgt_2,39,195,196).  visible(fcgt_2,196). happensAt(walking(fcgt_2),196).
orientation(fcgt_1,32,197). coords(fcgt_1,249,271,197).  visible(fcgt_1,197). happensAt(walking(fcgt_1),197). orientation(fcgt_2,166,197). coords(fcgt_2,39,194,197).  visible(fcgt_2,197). happensAt(walking(fcgt_2),197).
orientation(fcgt_1,28,198). coords(fcgt_1,248,270,198).  visible(fcgt_1,198). happensAt(walking(fcgt_1),198). orientation(fcgt_2,166,198). coords(fcgt_2,40,193,198).  visible(fcgt_2,198). happensAt(walking(fcgt_2),198).
orientation(fcgt_1,23,199). coords(fcgt_1,248,269,199).  visible(fcgt_1,199). happensAt(walking(fcgt_1),199). orientation(fcgt_2,166,199). coords(fcgt_2,40,193,199).  visible(fcgt_2,199). happensAt(walking(fcgt_2),199).
orientation(fcgt_1,23,200). coords(fcgt_1,246,268,200).  visible(fcgt_1,200). happensAt(walking(fcgt_1),200). orientation(fcgt_2,166,200). coords(fcgt_2,40,193,200).  visible(fcgt_2,200). happensAt(walking(fcgt_2),200).
orientation(fcgt_1,11,201). coords(fcgt_1,244,266,201).  visible(fcgt_1,201). happensAt(walking(fcgt_1),201). orientation(fcgt_2,166,201). coords(fcgt_2,40,192,201).  visible(fcgt_2,201). happensAt(walking(fcgt_2),201).
orientation(fcgt_1,7,202). coords(fcgt_1,241,264,202).  visible(fcgt_1,202). happensAt(walking(fcgt_1),202). orientation(fcgt_2,166,202). coords(fcgt_2,41,192,202).  visible(fcgt_2,202). happensAt(walking(fcgt_2),202).
orientation(fcgt_1,7,203). coords(fcgt_1,238,262,203).  visible(fcgt_1,203). happensAt(walking(fcgt_1),203). orientation(fcgt_2,166,203). coords(fcgt_2,41,191,203).  visible(fcgt_2,203). happensAt(walking(fcgt_2),203).
orientation(fcgt_1,2,204). coords(fcgt_1,236,261,204).  visible(fcgt_1,204). happensAt(walking(fcgt_1),204). orientation(fcgt_2,166,204). coords(fcgt_2,41,191,204).  visible(fcgt_2,204). happensAt(walking(fcgt_2),204).
orientation(fcgt_1,2,205). coords(fcgt_1,233,258,205).  visible(fcgt_1,205). happensAt(walking(fcgt_1),205). orientation(fcgt_2,166,205). coords(fcgt_2,42,190,205).  visible(fcgt_2,205). happensAt(walking(fcgt_2),205).
orientation(fcgt_1,2,206). coords(fcgt_1,231,255,206).  visible(fcgt_1,206). happensAt(walking(fcgt_1),206). orientation(fcgt_2,166,206). coords(fcgt_2,42,190,206).  visible(fcgt_2,206). happensAt(walking(fcgt_2),206).
orientation(fcgt_1,2,207). coords(fcgt_1,229,249,207).  visible(fcgt_1,207). happensAt(walking(fcgt_1),207). orientation(fcgt_2,166,207). coords(fcgt_2,42,190,207).  visible(fcgt_2,207). happensAt(walking(fcgt_2),207).
orientation(fcgt_1,2,208). coords(fcgt_1,229,244,208).  visible(fcgt_1,208). happensAt(walking(fcgt_1),208). orientation(fcgt_2,166,208). coords(fcgt_2,42,188,208).  visible(fcgt_2,208). happensAt(walking(fcgt_2),208).
orientation(fcgt_1,2,209). coords(fcgt_1,228,242,209).  visible(fcgt_1,209). happensAt(walking(fcgt_1),209). orientation(fcgt_2,158,209). coords(fcgt_2,42,188,209).  visible(fcgt_2,209). happensAt(walking(fcgt_2),209).
orientation(fcgt_1,2,210). coords(fcgt_1,227,240,210).  visible(fcgt_1,210). happensAt(walking(fcgt_1),210). orientation(fcgt_2,157,210). coords(fcgt_2,43,187,210).  visible(fcgt_2,210). happensAt(walking(fcgt_2),210).
orientation(fcgt_1,177,211). coords(fcgt_1,226,240,211).  visible(fcgt_1,211). happensAt(walking(fcgt_1),211). orientation(fcgt_2,157,211). coords(fcgt_2,43,185,211).  visible(fcgt_2,211). happensAt(walking(fcgt_2),211).
orientation(fcgt_1,177,212). coords(fcgt_1,224,238,212).  visible(fcgt_1,212). happensAt(walking(fcgt_1),212). orientation(fcgt_2,157,212). coords(fcgt_2,43,184,212).  visible(fcgt_2,212). happensAt(walking(fcgt_2),212).
orientation(fcgt_1,169,213). coords(fcgt_1,222,237,213).  visible(fcgt_1,213). happensAt(walking(fcgt_1),213). orientation(fcgt_2,157,213). coords(fcgt_2,44,184,213).  visible(fcgt_2,213). happensAt(walking(fcgt_2),213).
orientation(fcgt_1,161,214). coords(fcgt_1,219,235,214).  visible(fcgt_1,214). happensAt(walking(fcgt_1),214). orientation(fcgt_2,157,214). coords(fcgt_2,45,184,214).  visible(fcgt_2,214). happensAt(walking(fcgt_2),214).
orientation(fcgt_1,161,215). coords(fcgt_1,215,232,215).  visible(fcgt_1,215). happensAt(walking(fcgt_1),215). orientation(fcgt_2,157,215). coords(fcgt_2,46,184,215).  visible(fcgt_2,215). happensAt(walking(fcgt_2),215).
orientation(fcgt_1,161,216). coords(fcgt_1,214,227,216).  visible(fcgt_1,216). happensAt(walking(fcgt_1),216). orientation(fcgt_2,157,216). coords(fcgt_2,47,184,216).  visible(fcgt_2,216). happensAt(walking(fcgt_2),216).
orientation(fcgt_1,171,217). coords(fcgt_1,213,222,217).  visible(fcgt_1,217). happensAt(walking(fcgt_1),217). orientation(fcgt_2,157,217). coords(fcgt_2,47,183,217).  visible(fcgt_2,217). happensAt(walking(fcgt_2),217).
orientation(fcgt_1,171,218). coords(fcgt_1,212,219,218).  visible(fcgt_1,218). happensAt(walking(fcgt_1),218). orientation(fcgt_2,157,218). coords(fcgt_2,47,182,218).  visible(fcgt_2,218). happensAt(walking(fcgt_2),218).
orientation(fcgt_1,171,219). coords(fcgt_1,211,215,219).  visible(fcgt_1,219). happensAt(walking(fcgt_1),219). orientation(fcgt_2,157,219). coords(fcgt_2,47,181,219).  visible(fcgt_2,219). happensAt(walking(fcgt_2),219).
orientation(fcgt_1,171,220). coords(fcgt_1,210,212,220).  visible(fcgt_1,220). happensAt(walking(fcgt_1),220). orientation(fcgt_2,157,220). coords(fcgt_2,48,180,220).  visible(fcgt_2,220). happensAt(walking(fcgt_2),220).
#show holdsAt/2.