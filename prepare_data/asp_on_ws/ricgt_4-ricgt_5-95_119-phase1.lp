
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

	orientation(ricgt_4,172,95). coords(ricgt_4,26,134,95).  visible(ricgt_4,95). happensAt(walking(ricgt_4),95). orientation(ricgt_5,0,95). coords(ricgt_5,27,126,95).  visible(ricgt_5,95). happensAt(walking(ricgt_5),95).
orientation(ricgt_4,172,96). coords(ricgt_4,26,134,96).  visible(ricgt_4,96). happensAt(walking(ricgt_4),96). orientation(ricgt_5,0,96). coords(ricgt_5,27,126,96).  visible(ricgt_5,96). happensAt(walking(ricgt_5),96).
orientation(ricgt_4,172,97). coords(ricgt_4,26,134,97).  visible(ricgt_4,97). happensAt(walking(ricgt_4),97). orientation(ricgt_5,0,97). coords(ricgt_5,27,126,97).  visible(ricgt_5,97). happensAt(walking(ricgt_5),97).
orientation(ricgt_4,172,98). coords(ricgt_4,26,134,98).  visible(ricgt_4,98). happensAt(walking(ricgt_4),98). orientation(ricgt_5,0,98). coords(ricgt_5,27,126,98).  visible(ricgt_5,98). happensAt(walking(ricgt_5),98).
orientation(ricgt_4,172,99). coords(ricgt_4,26,134,99).  visible(ricgt_4,99). happensAt(walking(ricgt_4),99). orientation(ricgt_5,0,99). coords(ricgt_5,27,126,99).  visible(ricgt_5,99). happensAt(walking(ricgt_5),99).
orientation(ricgt_4,172,100). coords(ricgt_4,26,134,100).  visible(ricgt_4,100). happensAt(walking(ricgt_4),100). orientation(ricgt_5,0,100). coords(ricgt_5,27,126,100).  visible(ricgt_5,100). happensAt(walking(ricgt_5),100).
orientation(ricgt_4,172,101). coords(ricgt_4,26,134,101).  visible(ricgt_4,101). happensAt(walking(ricgt_4),101). orientation(ricgt_5,0,101). coords(ricgt_5,27,126,101).  visible(ricgt_5,101). happensAt(walking(ricgt_5),101).
orientation(ricgt_4,172,102). coords(ricgt_4,26,134,102).  visible(ricgt_4,102). happensAt(walking(ricgt_4),102). orientation(ricgt_5,0,102). coords(ricgt_5,27,126,102).  visible(ricgt_5,102). happensAt(walking(ricgt_5),102).
orientation(ricgt_4,172,103). coords(ricgt_4,26,134,103).  visible(ricgt_4,103). happensAt(walking(ricgt_4),103). orientation(ricgt_5,0,103). coords(ricgt_5,27,127,103).  visible(ricgt_5,103). happensAt(walking(ricgt_5),103).
orientation(ricgt_4,172,104). coords(ricgt_4,26,134,104).  visible(ricgt_4,104). happensAt(walking(ricgt_4),104). orientation(ricgt_5,0,104). coords(ricgt_5,27,127,104).  visible(ricgt_5,104). happensAt(walking(ricgt_5),104).
orientation(ricgt_4,172,105). coords(ricgt_4,26,134,105).  visible(ricgt_4,105). happensAt(walking(ricgt_4),105). orientation(ricgt_5,0,105). coords(ricgt_5,27,127,105).  visible(ricgt_5,105). happensAt(walking(ricgt_5),105).
orientation(ricgt_4,172,106). coords(ricgt_4,26,134,106).  visible(ricgt_4,106). happensAt(walking(ricgt_4),106). orientation(ricgt_5,0,106). coords(ricgt_5,27,127,106).  visible(ricgt_5,106). happensAt(walking(ricgt_5),106).
orientation(ricgt_4,172,107). coords(ricgt_4,26,134,107).  visible(ricgt_4,107). happensAt(walking(ricgt_4),107). orientation(ricgt_5,0,107). coords(ricgt_5,27,127,107).  visible(ricgt_5,107). happensAt(walking(ricgt_5),107).
orientation(ricgt_4,172,108). coords(ricgt_4,26,134,108).  visible(ricgt_4,108). happensAt(walking(ricgt_4),108). orientation(ricgt_5,0,108). coords(ricgt_5,27,127,108).  visible(ricgt_5,108). happensAt(walking(ricgt_5),108).
orientation(ricgt_4,172,109). coords(ricgt_4,26,134,109).  visible(ricgt_4,109). happensAt(walking(ricgt_4),109). orientation(ricgt_5,0,109). coords(ricgt_5,27,127,109).  visible(ricgt_5,109). happensAt(walking(ricgt_5),109).
orientation(ricgt_4,172,110). coords(ricgt_4,26,134,110).  visible(ricgt_4,110). happensAt(walking(ricgt_4),110). orientation(ricgt_5,0,110). coords(ricgt_5,27,127,110).  visible(ricgt_5,110). happensAt(walking(ricgt_5),110).
orientation(ricgt_4,172,111). coords(ricgt_4,26,134,111).  visible(ricgt_4,111). happensAt(walking(ricgt_4),111). orientation(ricgt_5,0,111). coords(ricgt_5,27,127,111).  visible(ricgt_5,111). happensAt(walking(ricgt_5),111).
orientation(ricgt_4,172,112). coords(ricgt_4,26,134,112).  visible(ricgt_4,112). happensAt(walking(ricgt_4),112). orientation(ricgt_5,0,112). coords(ricgt_5,27,127,112).  visible(ricgt_5,112). happensAt(walking(ricgt_5),112).
orientation(ricgt_4,172,113). coords(ricgt_4,26,134,113).  visible(ricgt_4,113). happensAt(walking(ricgt_4),113). orientation(ricgt_5,0,113). coords(ricgt_5,27,127,113).  visible(ricgt_5,113). happensAt(walking(ricgt_5),113).
orientation(ricgt_4,172,114). coords(ricgt_4,26,134,114).  visible(ricgt_4,114). happensAt(walking(ricgt_4),114). orientation(ricgt_5,0,114). coords(ricgt_5,27,127,114).  visible(ricgt_5,114). happensAt(walking(ricgt_5),114).
orientation(ricgt_4,172,115). coords(ricgt_4,26,134,115).  visible(ricgt_4,115). happensAt(walking(ricgt_4),115). orientation(ricgt_5,0,115). coords(ricgt_5,27,127,115).  visible(ricgt_5,115). happensAt(walking(ricgt_5),115).
orientation(ricgt_4,172,116). coords(ricgt_4,26,134,116).  visible(ricgt_4,116). happensAt(walking(ricgt_4),116). orientation(ricgt_5,0,116). coords(ricgt_5,27,127,116).  visible(ricgt_5,116). happensAt(walking(ricgt_5),116).
orientation(ricgt_4,172,117). coords(ricgt_4,26,134,117).  visible(ricgt_4,117). happensAt(walking(ricgt_4),117). orientation(ricgt_5,0,117). coords(ricgt_5,27,127,117).  visible(ricgt_5,117). happensAt(walking(ricgt_5),117).
orientation(ricgt_4,172,118). coords(ricgt_4,26,134,118).  visible(ricgt_4,118). happensAt(walking(ricgt_4),118). orientation(ricgt_5,0,118). coords(ricgt_5,27,127,118).  visible(ricgt_5,118). happensAt(walking(ricgt_5),118).
orientation(ricgt_4,172,119). coords(ricgt_4,26,134,119).  visible(ricgt_4,119). happensAt(walking(ricgt_4),119). orientation(ricgt_5,0,119). coords(ricgt_5,27,127,119).  visible(ricgt_5,119). happensAt(walking(ricgt_5),119).
#show holdsAt/2.