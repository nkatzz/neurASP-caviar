
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

	orientation(fcgt_0,150,36). coords(fcgt_0,178,133,36).  visible(fcgt_0,36). happensAt(walking(fcgt_0),36). orientation(fcgt_4,169,36). coords(fcgt_4,101,212,36).  visible(fcgt_4,36). happensAt(walking(fcgt_4),36).
orientation(fcgt_0,150,37). coords(fcgt_0,178,133,37).  visible(fcgt_0,37). happensAt(walking(fcgt_0),37). orientation(fcgt_4,169,37). coords(fcgt_4,101,212,37).  visible(fcgt_4,37). happensAt(walking(fcgt_4),37).
orientation(fcgt_0,150,38). coords(fcgt_0,178,133,38).  visible(fcgt_0,38). happensAt(walking(fcgt_0),38). orientation(fcgt_4,169,38). coords(fcgt_4,101,212,38).  visible(fcgt_4,38). happensAt(walking(fcgt_4),38).
orientation(fcgt_0,150,39). coords(fcgt_0,178,133,39).  visible(fcgt_0,39). happensAt(walking(fcgt_0),39). orientation(fcgt_4,169,39). coords(fcgt_4,101,212,39).  visible(fcgt_4,39). happensAt(walking(fcgt_4),39).
orientation(fcgt_0,143,40). coords(fcgt_0,179,133,40).  visible(fcgt_0,40). happensAt(walking(fcgt_0),40). orientation(fcgt_4,169,40). coords(fcgt_4,101,212,40).  visible(fcgt_4,40). happensAt(walking(fcgt_4),40).
orientation(fcgt_0,143,41). coords(fcgt_0,180,135,41).  visible(fcgt_0,41). happensAt(walking(fcgt_0),41). orientation(fcgt_4,169,41). coords(fcgt_4,101,211,41).  visible(fcgt_4,41). happensAt(walking(fcgt_4),41).
orientation(fcgt_0,143,42). coords(fcgt_0,180,137,42).  visible(fcgt_0,42). happensAt(walking(fcgt_0),42). orientation(fcgt_4,169,42). coords(fcgt_4,101,210,42).  visible(fcgt_4,42). happensAt(walking(fcgt_4),42).
orientation(fcgt_0,143,43). coords(fcgt_0,181,139,43).  visible(fcgt_0,43). happensAt(walking(fcgt_0),43). orientation(fcgt_4,169,43). coords(fcgt_4,101,210,43).  visible(fcgt_4,43). happensAt(walking(fcgt_4),43).
orientation(fcgt_0,143,44). coords(fcgt_0,184,141,44).  visible(fcgt_0,44). happensAt(walking(fcgt_0),44). orientation(fcgt_4,169,44). coords(fcgt_4,100,210,44).  visible(fcgt_4,44). happensAt(walking(fcgt_4),44).
orientation(fcgt_0,143,45). coords(fcgt_0,186,143,45).  visible(fcgt_0,45). happensAt(walking(fcgt_0),45). orientation(fcgt_4,169,45). coords(fcgt_4,100,210,45).  visible(fcgt_4,45). happensAt(walking(fcgt_4),45).
orientation(fcgt_0,143,46). coords(fcgt_0,187,144,46).  visible(fcgt_0,46). happensAt(walking(fcgt_0),46). orientation(fcgt_4,169,46). coords(fcgt_4,100,210,46).  visible(fcgt_4,46). happensAt(walking(fcgt_4),46).
orientation(fcgt_0,143,47). coords(fcgt_0,189,145,47).  visible(fcgt_0,47). happensAt(walking(fcgt_0),47). orientation(fcgt_4,169,47). coords(fcgt_4,100,210,47).  visible(fcgt_4,47). happensAt(walking(fcgt_4),47).
orientation(fcgt_0,143,48). coords(fcgt_0,190,145,48).  visible(fcgt_0,48). happensAt(walking(fcgt_0),48). orientation(fcgt_4,166,48). coords(fcgt_4,100,210,48).  visible(fcgt_4,48). happensAt(walking(fcgt_4),48).
orientation(fcgt_0,143,49). coords(fcgt_0,191,146,49).  visible(fcgt_0,49). happensAt(walking(fcgt_0),49). orientation(fcgt_4,166,49). coords(fcgt_4,100,210,49).  visible(fcgt_4,49). happensAt(walking(fcgt_4),49).
orientation(fcgt_0,143,50). coords(fcgt_0,192,147,50).  visible(fcgt_0,50). happensAt(walking(fcgt_0),50). orientation(fcgt_4,165,50). coords(fcgt_4,100,210,50).  visible(fcgt_4,50). happensAt(walking(fcgt_4),50).
orientation(fcgt_0,143,51). coords(fcgt_0,192,147,51).  visible(fcgt_0,51). happensAt(walking(fcgt_0),51). orientation(fcgt_4,165,51). coords(fcgt_4,100,209,51).  visible(fcgt_4,51). happensAt(inactive(fcgt_4),51).
orientation(fcgt_0,143,52). coords(fcgt_0,192,147,52).  visible(fcgt_0,52). happensAt(walking(fcgt_0),52). orientation(fcgt_4,165,52). coords(fcgt_4,100,209,52).  visible(fcgt_4,52). happensAt(inactive(fcgt_4),52).
orientation(fcgt_0,143,53). coords(fcgt_0,193,148,53).  visible(fcgt_0,53). happensAt(walking(fcgt_0),53). orientation(fcgt_4,165,53). coords(fcgt_4,100,209,53).  visible(fcgt_4,53). happensAt(inactive(fcgt_4),53).
orientation(fcgt_0,143,54). coords(fcgt_0,193,148,54).  visible(fcgt_0,54). happensAt(walking(fcgt_0),54). orientation(fcgt_4,165,54). coords(fcgt_4,100,209,54).  visible(fcgt_4,54). happensAt(inactive(fcgt_4),54).
orientation(fcgt_0,143,55). coords(fcgt_0,193,148,55).  visible(fcgt_0,55). happensAt(walking(fcgt_0),55). orientation(fcgt_4,165,55). coords(fcgt_4,100,209,55).  visible(fcgt_4,55). happensAt(inactive(fcgt_4),55).
orientation(fcgt_0,143,56). coords(fcgt_0,194,148,56).  visible(fcgt_0,56). happensAt(walking(fcgt_0),56). orientation(fcgt_4,165,56). coords(fcgt_4,100,209,56).  visible(fcgt_4,56). happensAt(inactive(fcgt_4),56).
orientation(fcgt_0,143,57). coords(fcgt_0,195,148,57).  visible(fcgt_0,57). happensAt(walking(fcgt_0),57). orientation(fcgt_4,165,57). coords(fcgt_4,100,209,57).  visible(fcgt_4,57). happensAt(inactive(fcgt_4),57).
orientation(fcgt_0,143,58). coords(fcgt_0,197,149,58).  visible(fcgt_0,58). happensAt(walking(fcgt_0),58). orientation(fcgt_4,165,58). coords(fcgt_4,100,208,58).  visible(fcgt_4,58). happensAt(inactive(fcgt_4),58).
orientation(fcgt_0,143,59). coords(fcgt_0,197,149,59).  visible(fcgt_0,59). happensAt(walking(fcgt_0),59). orientation(fcgt_4,165,59). coords(fcgt_4,100,208,59).  visible(fcgt_4,59). happensAt(inactive(fcgt_4),59).
orientation(fcgt_0,143,60). coords(fcgt_0,198,150,60).  visible(fcgt_0,60). happensAt(walking(fcgt_0),60). orientation(fcgt_4,165,60). coords(fcgt_4,100,208,60).  visible(fcgt_4,60). happensAt(inactive(fcgt_4),60).
#show holdsAt/2.