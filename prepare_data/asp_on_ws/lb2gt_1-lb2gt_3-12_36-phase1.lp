
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

	orientation(lb2gt_1,130,12). coords(lb2gt_1,214,60,12).  visible(lb2gt_1,12). happensAt(walking(lb2gt_1),12). orientation(lb2gt_3,146,12). coords(lb2gt_3,29,31,12).  visible(lb2gt_3,12). happensAt(walking(lb2gt_3),12).
orientation(lb2gt_1,130,13). coords(lb2gt_1,214,58,13).  visible(lb2gt_1,13). happensAt(walking(lb2gt_1),13). orientation(lb2gt_3,146,13). coords(lb2gt_3,29,30,13).  visible(lb2gt_3,13). happensAt(walking(lb2gt_3),13).
orientation(lb2gt_1,130,14). coords(lb2gt_1,215,55,14).  visible(lb2gt_1,14). happensAt(walking(lb2gt_1),14). orientation(lb2gt_3,146,14). coords(lb2gt_3,29,31,14).  visible(lb2gt_3,14). happensAt(walking(lb2gt_3),14).
orientation(lb2gt_1,130,15). coords(lb2gt_1,217,53,15).  visible(lb2gt_1,15). happensAt(walking(lb2gt_1),15). orientation(lb2gt_3,146,15). coords(lb2gt_3,29,31,15).  visible(lb2gt_3,15). happensAt(walking(lb2gt_3),15).
orientation(lb2gt_1,130,16). coords(lb2gt_1,219,53,16).  visible(lb2gt_1,16). happensAt(walking(lb2gt_1),16). orientation(lb2gt_3,146,16). coords(lb2gt_3,29,32,16).  visible(lb2gt_3,16). happensAt(walking(lb2gt_3),16).
orientation(lb2gt_1,130,17). coords(lb2gt_1,221,52,17).  visible(lb2gt_1,17). happensAt(walking(lb2gt_1),17). orientation(lb2gt_3,146,17). coords(lb2gt_3,27,32,17).  visible(lb2gt_3,17). happensAt(walking(lb2gt_3),17).
orientation(lb2gt_1,130,18). coords(lb2gt_1,224,51,18).  visible(lb2gt_1,18). happensAt(walking(lb2gt_1),18). orientation(lb2gt_3,146,18). coords(lb2gt_3,27,32,18).  visible(lb2gt_3,18). happensAt(walking(lb2gt_3),18).
orientation(lb2gt_1,130,19). coords(lb2gt_1,226,51,19).  visible(lb2gt_1,19). happensAt(walking(lb2gt_1),19). orientation(lb2gt_3,146,19). coords(lb2gt_3,26,33,19).  visible(lb2gt_3,19). happensAt(walking(lb2gt_3),19).
orientation(lb2gt_1,130,20). coords(lb2gt_1,227,50,20).  visible(lb2gt_1,20). happensAt(walking(lb2gt_1),20). orientation(lb2gt_3,146,20). coords(lb2gt_3,25,33,20).  visible(lb2gt_3,20). happensAt(walking(lb2gt_3),20).
orientation(lb2gt_1,130,21). coords(lb2gt_1,228,49,21).  visible(lb2gt_1,21). happensAt(walking(lb2gt_1),21). orientation(lb2gt_3,146,21). coords(lb2gt_3,25,33,21).  visible(lb2gt_3,21). happensAt(walking(lb2gt_3),21).
orientation(lb2gt_1,124,22). coords(lb2gt_1,229,47,22).  visible(lb2gt_1,22). happensAt(walking(lb2gt_1),22). orientation(lb2gt_3,146,22). coords(lb2gt_3,24,33,22).  visible(lb2gt_3,22). happensAt(walking(lb2gt_3),22).
orientation(lb2gt_1,121,23). coords(lb2gt_1,230,46,23).  visible(lb2gt_1,23). happensAt(walking(lb2gt_1),23). orientation(lb2gt_3,146,23). coords(lb2gt_3,24,34,23).  visible(lb2gt_3,23). happensAt(walking(lb2gt_3),23).
orientation(lb2gt_1,117,24). coords(lb2gt_1,231,44,24).  visible(lb2gt_1,24). happensAt(walking(lb2gt_1),24). orientation(lb2gt_3,146,24). coords(lb2gt_3,23,34,24).  visible(lb2gt_3,24). happensAt(walking(lb2gt_3),24).
orientation(lb2gt_1,112,25). coords(lb2gt_1,232,42,25).  visible(lb2gt_1,25). happensAt(walking(lb2gt_1),25). orientation(lb2gt_3,146,25). coords(lb2gt_3,22,35,25).  visible(lb2gt_3,25). happensAt(walking(lb2gt_3),25).
orientation(lb2gt_1,119,26). coords(lb2gt_1,232,39,26).  visible(lb2gt_1,26). happensAt(walking(lb2gt_1),26). orientation(lb2gt_3,146,26). coords(lb2gt_3,22,35,26).  visible(lb2gt_3,26). happensAt(walking(lb2gt_3),26).
orientation(lb2gt_1,117,27). coords(lb2gt_1,233,36,27).  visible(lb2gt_1,27). happensAt(walking(lb2gt_1),27). orientation(lb2gt_3,146,27). coords(lb2gt_3,22,35,27).  visible(lb2gt_3,27). happensAt(walking(lb2gt_3),27).
orientation(lb2gt_1,119,28). coords(lb2gt_1,233,34,28).  visible(lb2gt_1,28). happensAt(walking(lb2gt_1),28). orientation(lb2gt_3,146,28). coords(lb2gt_3,22,37,28).  visible(lb2gt_3,28). happensAt(walking(lb2gt_3),28).
orientation(lb2gt_1,115,29). coords(lb2gt_1,233,33,29).  visible(lb2gt_1,29). happensAt(walking(lb2gt_1),29). orientation(lb2gt_3,146,29). coords(lb2gt_3,23,37,29).  visible(lb2gt_3,29). happensAt(walking(lb2gt_3),29).
orientation(lb2gt_1,119,30). coords(lb2gt_1,234,33,30).  visible(lb2gt_1,30). happensAt(walking(lb2gt_1),30). orientation(lb2gt_3,146,30). coords(lb2gt_3,24,38,30).  visible(lb2gt_3,30). happensAt(walking(lb2gt_3),30).
orientation(lb2gt_1,118,31). coords(lb2gt_1,235,32,31).  visible(lb2gt_1,31). happensAt(walking(lb2gt_1),31). orientation(lb2gt_3,146,31). coords(lb2gt_3,24,39,31).  visible(lb2gt_3,31). happensAt(walking(lb2gt_3),31).
orientation(lb2gt_1,116,32). coords(lb2gt_1,236,32,32).  visible(lb2gt_1,32). happensAt(walking(lb2gt_1),32). orientation(lb2gt_3,146,32). coords(lb2gt_3,24,39,32).  visible(lb2gt_3,32). happensAt(walking(lb2gt_3),32).
orientation(lb2gt_1,116,33). coords(lb2gt_1,237,31,33).  visible(lb2gt_1,33). happensAt(walking(lb2gt_1),33). orientation(lb2gt_3,146,33). coords(lb2gt_3,24,39,33).  visible(lb2gt_3,33). happensAt(walking(lb2gt_3),33).
orientation(lb2gt_1,113,34). coords(lb2gt_1,238,30,34).  visible(lb2gt_1,34). happensAt(walking(lb2gt_1),34). orientation(lb2gt_3,146,34). coords(lb2gt_3,24,39,34).  visible(lb2gt_3,34). happensAt(walking(lb2gt_3),34).
orientation(lb2gt_1,110,35). coords(lb2gt_1,239,29,35).  visible(lb2gt_1,35). happensAt(walking(lb2gt_1),35). orientation(lb2gt_3,146,35). coords(lb2gt_3,23,40,35).  visible(lb2gt_3,35). happensAt(walking(lb2gt_3),35).
orientation(lb2gt_1,109,36). coords(lb2gt_1,240,28,36).  visible(lb2gt_1,36). happensAt(walking(lb2gt_1),36). orientation(lb2gt_3,146,36). coords(lb2gt_3,23,40,36).  visible(lb2gt_3,36). happensAt(walking(lb2gt_3),36).
#show holdsAt/2.