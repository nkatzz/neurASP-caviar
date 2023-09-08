
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

	orientation(fcgt_2,166,198). coords(fcgt_2,40,193,198).  visible(fcgt_2,198). happensAt(walking(fcgt_2),198). orientation(fcgt_3,159,198). coords(fcgt_3,103,163,198).  visible(fcgt_3,198). happensAt(inactive(fcgt_3),198).
orientation(fcgt_2,166,199). coords(fcgt_2,40,193,199).  visible(fcgt_2,199). happensAt(walking(fcgt_2),199). orientation(fcgt_3,159,199). coords(fcgt_3,103,163,199).  visible(fcgt_3,199). happensAt(inactive(fcgt_3),199).
orientation(fcgt_2,166,200). coords(fcgt_2,40,193,200).  visible(fcgt_2,200). happensAt(walking(fcgt_2),200). orientation(fcgt_3,159,200). coords(fcgt_3,103,163,200).  visible(fcgt_3,200). happensAt(inactive(fcgt_3),200).
orientation(fcgt_2,166,201). coords(fcgt_2,40,192,201).  visible(fcgt_2,201). happensAt(walking(fcgt_2),201). orientation(fcgt_3,159,201). coords(fcgt_3,102,163,201).  visible(fcgt_3,201). happensAt(inactive(fcgt_3),201).
orientation(fcgt_2,166,202). coords(fcgt_2,41,192,202).  visible(fcgt_2,202). happensAt(walking(fcgt_2),202). orientation(fcgt_3,159,202). coords(fcgt_3,102,163,202).  visible(fcgt_3,202). happensAt(walking(fcgt_3),202).
orientation(fcgt_2,166,203). coords(fcgt_2,41,191,203).  visible(fcgt_2,203). happensAt(walking(fcgt_2),203). orientation(fcgt_3,159,203). coords(fcgt_3,101,163,203).  visible(fcgt_3,203). happensAt(walking(fcgt_3),203).
orientation(fcgt_2,166,204). coords(fcgt_2,41,191,204).  visible(fcgt_2,204). happensAt(walking(fcgt_2),204). orientation(fcgt_3,159,204). coords(fcgt_3,101,164,204).  visible(fcgt_3,204). happensAt(walking(fcgt_3),204).
orientation(fcgt_2,166,205). coords(fcgt_2,42,190,205).  visible(fcgt_2,205). happensAt(walking(fcgt_2),205). orientation(fcgt_3,159,205). coords(fcgt_3,101,164,205).  visible(fcgt_3,205). happensAt(walking(fcgt_3),205).
orientation(fcgt_2,166,206). coords(fcgt_2,42,190,206).  visible(fcgt_2,206). happensAt(walking(fcgt_2),206). orientation(fcgt_3,159,206). coords(fcgt_3,101,165,206).  visible(fcgt_3,206). happensAt(walking(fcgt_3),206).
orientation(fcgt_2,166,207). coords(fcgt_2,42,190,207).  visible(fcgt_2,207). happensAt(walking(fcgt_2),207). orientation(fcgt_3,159,207). coords(fcgt_3,101,165,207).  visible(fcgt_3,207). happensAt(walking(fcgt_3),207).
orientation(fcgt_2,166,208). coords(fcgt_2,42,188,208).  visible(fcgt_2,208). happensAt(walking(fcgt_2),208). orientation(fcgt_3,153,208). coords(fcgt_3,100,166,208).  visible(fcgt_3,208). happensAt(walking(fcgt_3),208).
orientation(fcgt_2,158,209). coords(fcgt_2,42,188,209).  visible(fcgt_2,209). happensAt(walking(fcgt_2),209). orientation(fcgt_3,153,209). coords(fcgt_3,100,166,209).  visible(fcgt_3,209). happensAt(walking(fcgt_3),209).
orientation(fcgt_2,157,210). coords(fcgt_2,43,187,210).  visible(fcgt_2,210). happensAt(walking(fcgt_2),210). orientation(fcgt_3,149,210). coords(fcgt_3,100,166,210).  visible(fcgt_3,210). happensAt(walking(fcgt_3),210).
orientation(fcgt_2,157,211). coords(fcgt_2,43,185,211).  visible(fcgt_2,211). happensAt(walking(fcgt_2),211). orientation(fcgt_3,149,211). coords(fcgt_3,100,167,211).  visible(fcgt_3,211). happensAt(walking(fcgt_3),211).
orientation(fcgt_2,157,212). coords(fcgt_2,43,184,212).  visible(fcgt_2,212). happensAt(walking(fcgt_2),212). orientation(fcgt_3,149,212). coords(fcgt_3,100,167,212).  visible(fcgt_3,212). happensAt(walking(fcgt_3),212).
orientation(fcgt_2,157,213). coords(fcgt_2,44,184,213).  visible(fcgt_2,213). happensAt(walking(fcgt_2),213). orientation(fcgt_3,149,213). coords(fcgt_3,100,168,213).  visible(fcgt_3,213). happensAt(walking(fcgt_3),213).
orientation(fcgt_2,157,214). coords(fcgt_2,45,184,214).  visible(fcgt_2,214). happensAt(walking(fcgt_2),214). orientation(fcgt_3,149,214). coords(fcgt_3,100,168,214).  visible(fcgt_3,214). happensAt(walking(fcgt_3),214).
orientation(fcgt_2,157,215). coords(fcgt_2,46,184,215).  visible(fcgt_2,215). happensAt(walking(fcgt_2),215). orientation(fcgt_3,149,215). coords(fcgt_3,100,168,215).  visible(fcgt_3,215). happensAt(walking(fcgt_3),215).
orientation(fcgt_2,157,216). coords(fcgt_2,47,184,216).  visible(fcgt_2,216). happensAt(walking(fcgt_2),216). orientation(fcgt_3,149,216). coords(fcgt_3,100,168,216).  visible(fcgt_3,216). happensAt(walking(fcgt_3),216).
orientation(fcgt_2,157,217). coords(fcgt_2,47,183,217).  visible(fcgt_2,217). happensAt(walking(fcgt_2),217). orientation(fcgt_3,149,217). coords(fcgt_3,100,169,217).  visible(fcgt_3,217). happensAt(walking(fcgt_3),217).
orientation(fcgt_2,157,218). coords(fcgt_2,47,182,218).  visible(fcgt_2,218). happensAt(walking(fcgt_2),218). orientation(fcgt_3,149,218). coords(fcgt_3,100,169,218).  visible(fcgt_3,218). happensAt(walking(fcgt_3),218).
orientation(fcgt_2,157,219). coords(fcgt_2,47,181,219).  visible(fcgt_2,219). happensAt(walking(fcgt_2),219). orientation(fcgt_3,149,219). coords(fcgt_3,100,169,219).  visible(fcgt_3,219). happensAt(walking(fcgt_3),219).
orientation(fcgt_2,157,220). coords(fcgt_2,48,180,220).  visible(fcgt_2,220). happensAt(walking(fcgt_2),220). orientation(fcgt_3,149,220). coords(fcgt_3,100,170,220).  visible(fcgt_3,220). happensAt(walking(fcgt_3),220).
orientation(fcgt_2,157,221). coords(fcgt_2,48,179,221).  visible(fcgt_2,221). happensAt(walking(fcgt_2),221). orientation(fcgt_3,149,221). coords(fcgt_3,100,170,221).  visible(fcgt_3,221). happensAt(walking(fcgt_3),221).
orientation(fcgt_2,157,222). coords(fcgt_2,48,178,222).  visible(fcgt_2,222). happensAt(walking(fcgt_2),222). orientation(fcgt_3,149,222). coords(fcgt_3,100,170,222).  visible(fcgt_3,222). happensAt(walking(fcgt_3),222).
#show holdsAt/2.