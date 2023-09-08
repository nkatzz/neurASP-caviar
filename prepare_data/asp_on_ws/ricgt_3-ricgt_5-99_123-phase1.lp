
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

	orientation(ricgt_3,155,99). coords(ricgt_3,212,190,99).  visible(ricgt_3,99). happensAt(walking(ricgt_3),99). orientation(ricgt_5,0,99). coords(ricgt_5,27,126,99).  visible(ricgt_5,99). happensAt(walking(ricgt_5),99).
orientation(ricgt_3,158,100). coords(ricgt_3,212,188,100).  visible(ricgt_3,100). happensAt(walking(ricgt_3),100). orientation(ricgt_5,0,100). coords(ricgt_5,27,126,100).  visible(ricgt_5,100). happensAt(walking(ricgt_5),100).
orientation(ricgt_3,159,101). coords(ricgt_3,209,186,101).  visible(ricgt_3,101). happensAt(walking(ricgt_3),101). orientation(ricgt_5,0,101). coords(ricgt_5,27,126,101).  visible(ricgt_5,101). happensAt(walking(ricgt_5),101).
orientation(ricgt_3,161,102). coords(ricgt_3,208,184,102).  visible(ricgt_3,102). happensAt(walking(ricgt_3),102). orientation(ricgt_5,0,102). coords(ricgt_5,27,126,102).  visible(ricgt_5,102). happensAt(walking(ricgt_5),102).
orientation(ricgt_3,163,103). coords(ricgt_3,206,181,103).  visible(ricgt_3,103). happensAt(walking(ricgt_3),103). orientation(ricgt_5,0,103). coords(ricgt_5,27,127,103).  visible(ricgt_5,103). happensAt(walking(ricgt_5),103).
orientation(ricgt_3,161,104). coords(ricgt_3,202,179,104).  visible(ricgt_3,104). happensAt(walking(ricgt_3),104). orientation(ricgt_5,0,104). coords(ricgt_5,27,127,104).  visible(ricgt_5,104). happensAt(walking(ricgt_5),104).
orientation(ricgt_3,159,105). coords(ricgt_3,201,176,105).  visible(ricgt_3,105). happensAt(walking(ricgt_3),105). orientation(ricgt_5,0,105). coords(ricgt_5,27,127,105).  visible(ricgt_5,105). happensAt(walking(ricgt_5),105).
orientation(ricgt_3,158,106). coords(ricgt_3,201,175,106).  visible(ricgt_3,106). happensAt(walking(ricgt_3),106). orientation(ricgt_5,0,106). coords(ricgt_5,27,127,106).  visible(ricgt_5,106). happensAt(walking(ricgt_5),106).
orientation(ricgt_3,157,107). coords(ricgt_3,198,174,107).  visible(ricgt_3,107). happensAt(walking(ricgt_3),107). orientation(ricgt_5,0,107). coords(ricgt_5,27,127,107).  visible(ricgt_5,107). happensAt(walking(ricgt_5),107).
orientation(ricgt_3,155,108). coords(ricgt_3,197,173,108).  visible(ricgt_3,108). happensAt(walking(ricgt_3),108). orientation(ricgt_5,0,108). coords(ricgt_5,27,127,108).  visible(ricgt_5,108). happensAt(walking(ricgt_5),108).
orientation(ricgt_3,155,109). coords(ricgt_3,196,172,109).  visible(ricgt_3,109). happensAt(walking(ricgt_3),109). orientation(ricgt_5,0,109). coords(ricgt_5,27,127,109).  visible(ricgt_5,109). happensAt(walking(ricgt_5),109).
orientation(ricgt_3,155,110). coords(ricgt_3,194,171,110).  visible(ricgt_3,110). happensAt(walking(ricgt_3),110). orientation(ricgt_5,0,110). coords(ricgt_5,27,127,110).  visible(ricgt_5,110). happensAt(walking(ricgt_5),110).
orientation(ricgt_3,153,111). coords(ricgt_3,192,170,111).  visible(ricgt_3,111). happensAt(walking(ricgt_3),111). orientation(ricgt_5,0,111). coords(ricgt_5,27,127,111).  visible(ricgt_5,111). happensAt(walking(ricgt_5),111).
orientation(ricgt_3,151,112). coords(ricgt_3,192,170,112).  visible(ricgt_3,112). happensAt(walking(ricgt_3),112). orientation(ricgt_5,0,112). coords(ricgt_5,27,127,112).  visible(ricgt_5,112). happensAt(walking(ricgt_5),112).
orientation(ricgt_3,150,113). coords(ricgt_3,192,169,113).  visible(ricgt_3,113). happensAt(walking(ricgt_3),113). orientation(ricgt_5,0,113). coords(ricgt_5,27,127,113).  visible(ricgt_5,113). happensAt(walking(ricgt_5),113).
orientation(ricgt_3,150,114). coords(ricgt_3,192,168,114).  visible(ricgt_3,114). happensAt(walking(ricgt_3),114). orientation(ricgt_5,0,114). coords(ricgt_5,27,127,114).  visible(ricgt_5,114). happensAt(walking(ricgt_5),114).
orientation(ricgt_3,150,115). coords(ricgt_3,192,167,115).  visible(ricgt_3,115). happensAt(walking(ricgt_3),115). orientation(ricgt_5,0,115). coords(ricgt_5,27,127,115).  visible(ricgt_5,115). happensAt(walking(ricgt_5),115).
orientation(ricgt_3,150,116). coords(ricgt_3,191,167,116).  visible(ricgt_3,116). happensAt(walking(ricgt_3),116). orientation(ricgt_5,0,116). coords(ricgt_5,27,127,116).  visible(ricgt_5,116). happensAt(walking(ricgt_5),116).
orientation(ricgt_3,150,117). coords(ricgt_3,191,165,117).  visible(ricgt_3,117). happensAt(walking(ricgt_3),117). orientation(ricgt_5,0,117). coords(ricgt_5,27,127,117).  visible(ricgt_5,117). happensAt(walking(ricgt_5),117).
orientation(ricgt_3,148,118). coords(ricgt_3,191,165,118).  visible(ricgt_3,118). happensAt(walking(ricgt_3),118). orientation(ricgt_5,0,118). coords(ricgt_5,27,127,118).  visible(ricgt_5,118). happensAt(walking(ricgt_5),118).
orientation(ricgt_3,147,119). coords(ricgt_3,191,163,119).  visible(ricgt_3,119). happensAt(walking(ricgt_3),119). orientation(ricgt_5,0,119). coords(ricgt_5,27,127,119).  visible(ricgt_5,119). happensAt(walking(ricgt_5),119).
orientation(ricgt_3,148,120). coords(ricgt_3,191,162,120).  visible(ricgt_3,120). happensAt(walking(ricgt_3),120). orientation(ricgt_5,0,120). coords(ricgt_5,27,127,120).  visible(ricgt_5,120). happensAt(walking(ricgt_5),120).
orientation(ricgt_3,149,121). coords(ricgt_3,191,160,121).  visible(ricgt_3,121). happensAt(walking(ricgt_3),121). orientation(ricgt_5,0,121). coords(ricgt_5,27,127,121).  visible(ricgt_5,121). happensAt(walking(ricgt_5),121).
orientation(ricgt_3,151,122). coords(ricgt_3,190,159,122).  visible(ricgt_3,122). happensAt(walking(ricgt_3),122). orientation(ricgt_5,0,122). coords(ricgt_5,27,127,122).  visible(ricgt_5,122). happensAt(walking(ricgt_5),122).
orientation(ricgt_3,152,123). coords(ricgt_3,190,158,123).  visible(ricgt_3,123). happensAt(walking(ricgt_3),123). orientation(ricgt_5,0,123). coords(ricgt_5,27,127,123).  visible(ricgt_5,123). happensAt(walking(ricgt_5),123).
#show holdsAt/2.